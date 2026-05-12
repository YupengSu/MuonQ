## Muon code from Moonlight
## https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
import torch
from functools import partial
import math
import warnings
from .polar_method import jiacheng, zeropower_via_newtonschulz5, svd_exact_polar, PolarExpress, FastApplyPolarExpress, damped_zeropower_via_newtonschulz5
from ..quantize import quantize_sym, dequantize_sym

@torch.compile
def poweriter1(B, Q):
    P = B @ Q
    P, _ = torch.linalg.qr(P, mode='reduced')
    R = B.mT @ P
    return P, R

class MuonQ(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self,
                 named_params,
                 lr=1e-3,
                 weight_decay=0.1,
                 momentum=0.95,
                 nesterov=False,
                 ns_steps=5,
                 ns_damping=0.0,
                 rms_scaling=False,
                 nuclear_scaling=False,
                 polar_method="Keller",
                 adamw_betas=(0.95, 0.95),
                 adamw_eps=1e-8,
                 split_qkv=True,
                 polar_args={},
                 nheads=None,
                 qbit = 4,
                 gran = "tensor", 
                 norm = False,
                 compand = False,
                 rank = 0,
                ):
        """
        Arguments:
            polar_method: The name of the polar factorization method to use (e.g., "NewtonSchultz", "Keller", "Pole") where PolE = PolarExpress
        """
        defaults = dict(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                rms_scaling=rms_scaling,
                nuclear_scaling=nuclear_scaling,
                adamw_betas=adamw_betas,
                adamw_eps=adamw_eps,
                qbit=qbit,
                gran=gran,
                norm=norm,
                compand=compand,
                rank=rank,
        )
        
        # print("EMBED TOKENS AND LM_HEAD ARE NOT HANDLED CORRECTLY FOR MUON, THEY SHOULD BE WITH ADAMW.")
        muon_params, muon_params_names = [], []
        adamw_params, adamw_params_names = [], []
        for name, p in named_params:
            if p.ndim >= 2 and not any(excluded in name for excluded in ["embeddings", "embed_tokens", "wte", "lm_head", "wpe"]):
                muon_params.append(p)
                muon_params_names.append(name)
            else:
                adamw_params.append(p)
                adamw_params_names.append(name)
        params = list(muon_params)
        params.extend(adamw_params)
        self.split_qkv = split_qkv
        self.ns_damping = ns_damping
        print("Muon Quantization bits:", qbit, "Granularity:", gran, "Normalization:", norm, "Companding:", compand)
        super().__init__(params, defaults)
        
        # Sort parameters into those for which we will use Muon, and those for which we will not
# Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
        for p, p_name in zip(muon_params, muon_params_names):
            if not self.split_qkv: assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
            self.state[p]["param_name"] = p_name
            if p_name.endswith("attn.c_attn.weight"):
                self.state[p]["is_W_QKV"] = True
            elif p_name.endswith("attn.c_proj.weight"):
                self.state[p]["is_W_O"] = True

        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

        # Instantiate the polar factorization method
        self.polar_factorizer = self._initialize_polar_factorizer(polar_method, polar_args)

    def _initialize_polar_factorizer(self, polar_method, polar_args):
        """Initialize the polar factorization method based on the provided name and parameters."""
        if polar_method == "Keller":
            return zeropower_via_newtonschulz5  # Use the method directly
        elif polar_method == "DampedNS":
            return partial(damped_zeropower_via_newtonschulz5, damping=self.ns_damping)
        elif polar_method == "Jiacheng":
            return jiacheng
        elif polar_method == "polarexpress":
            return PolarExpress 
        elif polar_method == "fast_polarexpress":
            return partial(FastApplyPolarExpress, restart_interval=3, shift_eps=1e-3)
        elif polar_method == "svd-exact":
            return partial(svd_exact_polar, cutoff=polar_args.get("svd_cutoff", None), reverse=polar_args.get("svd_reverse", False))
        else:
            raise ValueError(f"Unknown polar method: {polar_method}")

    def adjust_lr_for_muon(self, lr, rms_scaling, nuclear_scaling, param_shape, grad, grad_sign):
        A, B = param_shape[:2]
        scale = 0.2 * math.sqrt(max(A, B))
        if rms_scaling:
            fan_out, fan_in = param_shape[:2]
            scale *= math.sqrt(fan_out / fan_in)
        if nuclear_scaling:
            scale *= torch.trace(grad.T @ grad_sign)
        return lr * scale

    def step(self, closure=None):
        """Perform a single optimization step.
        
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                        
        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            qbit = group["qbit"]
            gran = group["gran"]
            norm = group["norm"]
            compand = group["compand"]
            rank = group["rank"]

            # generate weight updates in distributed fashion
            for p in params:
                g = p.grad
                if g is None:
                    continue
                if (g.ndim > 2) and not (self.split_qkv):
                    g = g.view(g.size(0), -1)

                assert g is not None
                
                if self.split_qkv and self.state[p].get("is_W_QKV", False):
                    # we split qkv into separate q, k, v
                    if g.shape[0] % 3 != 0:
                        raise ValueError(f"Expected fused QKV shape with dim0 divisible by 3, got {g.shape}")
                    g = g.reshape(3 , g.shape[0] // 3 , g.shape[1])
                else:
                    g = g.unsqueeze(0)  # add a fake batch dimension for uniform processing
                
                # calc update
                state = self.state[p]
                
                if "momentum_res" not in state:
                    buf = torch.zeros_like(g)
                    if rank != 0:
                        R = torch.randn(buf.shape[0], buf.shape[-1], min(g.shape[-1], g.shape[-2]) // rank, device=buf.device)
                else:
                    buf_res = dequantize_sym(state["momentum_res"],
                                            state["momentum_res_scale"],
                                            state["momentum_res_meta"],)
                    if rank != 0:
                        P = dequantize_sym(state["momentum_P"],
                                            state["momentum_P_scale"],
                                            state["momentum_P_meta"],)
                        R = dequantize_sym(state["momentum_R"],
                                            state["momentum_R_scale"],
                                            state["momentum_R_meta"],)
                        buf = buf_res + P @ R.mT
                    else:
                        buf = buf_res
                

                g = g / (torch.norm(g) + 1e-7) if norm else g
                buf.mul_(momentum).add_(g)
                buf = buf / (torch.norm(buf) + 1e-7) if norm else buf
                
                if rank != 0:
                    Q = R / (torch.norm(R, dim=-2, keepdim=True) + 1e-7)
                    P, R = poweriter1(buf, Q)
                    res = buf - P @ R.mT
                        
                else:
                    res = buf

                state["momentum_res"], state["momentum_res_scale"], state["momentum_res_meta"] = \
                    quantize_sym(res, num_bits=qbit, granularity=gran, compand=compand)

                if rank != 0:
                    state["momentum_P"], state["momentum_P_scale"], state["momentum_P_meta"] = \
                        quantize_sym(P, num_bits=qbit, granularity="col", compand=compand)
                    state["momentum_R"], state["momentum_R_scale"], state["momentum_R_meta"] = \
                        quantize_sym(R, num_bits=qbit, granularity="col", compand=compand)

                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Use the selected polar factorization method
                u = self.polar_factorizer(g, group["ns_steps"])
                
                if self.split_qkv and self.state[p].get("is_W_QKV", False):
                    # recombine qkv
                    shape = g.shape[1:]
                    g = g.reshape(3 * g.shape[1], g.shape[2])
                    u = u.reshape(3 * u.shape[1], u.shape[2])
                else:
                    shape = g.shape[1:]
                    g = g.squeeze(0)
                    u = u.squeeze(0)
                
                # scale update
                adjusted_lr = self.adjust_lr_for_muon(
                    lr,
                    group["rms_scaling"],
                    group["nuclear_scaling"],
                    shape,
                    g.bfloat16(),  # convert to float16 to be compatible with u
                    u
                )

                # apply weight decay
                p.data.mul_(1 - lr * weight_decay)
                
                # apply update
                p.data.add_(u, alpha=-adjusted_lr)
                
            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["weight_decay"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)
                    
        return loss

