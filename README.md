# MuonQ: Enhancing Low-Bit Muon Quantization via Directional Fidelity Optimization

MuonQ is a low-bit training framework for the Muon optimizer. This repository contains the training code, Hydra configs, and data preprocessing
pipeline used for MuonQ experiments.

## Highlights

- **Pure 4-bit Muon state quantization** for matrix-shaped hidden-layer parameters.
- **Directional fidelity optimization** through three components:
  - pre-quantization normalization,
  - structural decomposition via power iteration,
  - mu-law companding quantization.
- **Memory efficient training**: MuonQ reduces optimizer-state memory by up to
  **7.3x** while closely matching full-precision Muon in training loss and
  downstream zero-shot accuracy in the paper experiments.
- **Hydra-based experiments** for GPT-style and LLaMA-style language models.

## Environment Setup

Create and activate a Python environment. Python 3.10 or newer is recommended.

```bash
conda create -n muonq python=3.12 -y
conda activate muonq
```

Install PyTorch separately so the CUDA build matches your machine. For example,
for CUDA 12.8:

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Create the required local environment file:

```bash
cp .env.example .env
```

Edit `.env` for your machine:

```bash
DATA_DIR=/path/to/preprocessed/data
HF_HOME=/path/to/huggingface/cache
```

`DATA_DIR` is where preprocessed binary dataset shards are written and read.
`HF_HOME` is the Hugging Face cache directory.

## Data Preparation

After setting up `.env`, preprocess the default FineWeb-100B dataset with the
LLaMA tokenizer:

```bash
python process_data.py --name fineweb100B --tokenizer llama2
```

This writes binary shards under:

```text
$DATA_DIR/fineweb100B-Llama-2-7b-hf/
```

That path is used by the LLaMA recipes in `hydra_conf/recipe/`.

## Training

Training is configured through Hydra. A typical launch command is:

```bash
GPUS=0,1,2,3
NGPUS=4
RECIPE=llama-60m
OPT=muonq
RUN_NAME=${RECIPE}_${OPT}

CUDA_VISIBLE_DEVICES=$GPUS \
torchrun \
  --standalone \
  --nproc-per-node=$NGPUS \
  run_hydra.py -cn test_hydra \
    recipe=${RECIPE} \
    optimizer_params=${OPT} \
    +logging_params.wandb.project=MuonQ \
    +logging_params.wandb.name=${RUN_NAME} \
  |& tee logs/${RUN_NAME}.log
```

`recipe=${RECIPE}` selects a config from `hydra_conf/recipe/`.

`optimizer_params=${OPT}` selects a config from `hydra_conf/optimizer_params/`.

If you use Weights & Biases logging, log in before training:

```bash
wandb login
```

`run.sh` is a minimal wrapper around the same command:

```bash
bash run.sh 0,1,2,3 muonq llama-60m
```


## Citation

The paper is currently under double-blind review. Citation information will be
added after release.
