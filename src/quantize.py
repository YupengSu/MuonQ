import math
import torch
from typing import Tuple, Optional

_MU = 255.0
_LOG1P_MU = math.log1p(_MU)
_INV_LOG1P_MU = 1.0 / _LOG1P_MU
_INV_MU = 1.0 / _MU

def mu_law_compress(x):
    x = x.clamp(-1.0, 1.0)
    ax = x.abs()
    return x.sign() * torch.log1p(_MU * ax) * _INV_LOG1P_MU

def mu_law_expand(y):
    ay = y.abs()
    return y.sign() * torch.expm1(ay * _LOG1P_MU) * _INV_MU

def stochastic_round(z: torch.Tensor):
    floor = torch.floor(z)
    prob = z - floor
    rnd = torch.rand_like(z)
    return floor + (rnd < prob).float()


def _check_num_bits(num_bits: int):
    if num_bits not in (2, 4, 8):
        raise ValueError(f"num_bits must be one of {{2, 4, 8}}, got {num_bits}")


def _pack_lowbit(q: torch.Tensor, num_bits: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    q: int tensor, value range:
       2-bit: [-2, 1]
       4-bit: [-8, 7]
       8-bit: [-128, 127]
    return:
       packed: uint8
       original_shape
    """
    _check_num_bits(num_bits)

    original_shape = tuple(q.shape)
    q = q.to(torch.int16)

    if num_bits == 8:
        return q.to(torch.int8).view(torch.uint8), original_shape

    offset = 1 << (num_bits - 1)          # 2bit->2, 4bit->8
    q_unsigned = (q + offset).to(torch.uint8).reshape(-1)

    vals_per_byte = 8 // num_bits
    n = q_unsigned.numel()
    packed_len = (n + vals_per_byte - 1) // vals_per_byte

    if packed_len * vals_per_byte != n:
        pad = packed_len * vals_per_byte - n
        q_unsigned = torch.cat(
            [q_unsigned, torch.zeros(pad, dtype=torch.uint8, device=q.device)],
            dim=0
        )

    q_unsigned = q_unsigned.view(packed_len, vals_per_byte)

    if num_bits == 4:
        packed = q_unsigned[:, 0] | (q_unsigned[:, 1] << 4)

    elif num_bits == 2:
        packed = (
            q_unsigned[:, 0]
            | (q_unsigned[:, 1] << 2)
            | (q_unsigned[:, 2] << 4)
            | (q_unsigned[:, 3] << 6)
        )

    else:
        raise RuntimeError("unreachable")

    return packed, original_shape


def _unpack_lowbit(
    packed: torch.Tensor,
    num_bits: int,
    original_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    _check_num_bits(num_bits)

    if device is None:
        device = packed.device

    total_elems = math.prod(original_shape)

    if num_bits == 8:
        q = packed.view(torch.int8).reshape(original_shape)
        return q

    packed = packed.to(torch.uint8).reshape(-1)

    if num_bits == 4:
        v0 = packed & 0x0F
        v1 = (packed >> 4) & 0x0F
        q_unsigned = torch.stack([v0, v1], dim=1).reshape(-1)

    elif num_bits == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        q_unsigned = torch.stack([v0, v1, v2, v3], dim=1).reshape(-1)

    else:
        raise RuntimeError("unreachable")

    q_unsigned = q_unsigned[:total_elems]
    offset = 1 << (num_bits - 1)
    q = q_unsigned.to(torch.int16) - offset
    return q.reshape(original_shape).to(torch.int8)

def quantize_sym(
    x: torch.Tensor,
    num_bits: int = 4,
    granularity: str = "tensor",   # "tensor" | "row" | "col"
    eps: float = 1e-8,
    stochastic: bool = False,
    compand: bool = False,
    scale_dtype: torch.dtype = torch.float16,
):
    assert x.dim() == 3  # (B, m, n)
    _check_num_bits(num_bits)
    assert granularity in {"tensor", "row", "col"}

    qmax = 2 ** (num_bits - 1) - 1
    qmin = - (2 ** (num_bits - 1))

    round_fn = stochastic_round if stochastic else torch.round

    x_work = mu_law_compress(x) if compand else x

    if granularity == "tensor":
        max_val = x_work.abs().amax(dim=(1, 2), keepdim=True)    # (B,1,1)
        scale = (max_val / qmax).clamp(min=eps)

    elif granularity == "row":
        max_val = x_work.abs().amax(dim=2, keepdim=True)         # (B,m,1)
        scale = (max_val / qmax).clamp(min=eps)

    elif granularity == "col":
        max_val = x_work.abs().amax(dim=1, keepdim=True)         # (B,1,n)
        scale = (max_val / qmax).clamp(min=eps)

    q = round_fn(x_work / scale).clamp(qmin, qmax).to(torch.int8)

    packed_q, x_shape = _pack_lowbit(q, num_bits)
    scale = scale.to(scale_dtype)

    meta = {
        "num_bits": num_bits,
        "granularity": granularity,
        "compand": compand,
        "x_shape": x_shape,
    }
    return packed_q, scale, meta


def dequantize_sym(
    packed_q: torch.Tensor,
    scale: torch.Tensor,
    meta: dict,
) -> torch.Tensor:
    num_bits = meta["num_bits"]
    compand = meta["compand"]
    x_shape = tuple(meta["x_shape"])

    q = _unpack_lowbit(packed_q, num_bits=num_bits, original_shape=x_shape)
    x = q.float() * scale.float()

    if compand:
        x = mu_law_expand(x)

    return x