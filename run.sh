#!/bin/bash
set -Eeuo pipefail

GPUS=${1:-4,5,6,7}
OPT=${2:-muonq}
RECIPE=${3:-llama-60m}
RUN_NAME="${RECIPE}_${OPT}"
export CUDA_VISIBLE_DEVICES=$GPUS
NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

mkdir -p logs

torchrun \
  --standalone \
  --nproc-per-node=$NGPUS \
  run_hydra.py -cn test_hydra \
    recipe=${RECIPE} \
    optimizer_params=${OPT} \
    +logging_params.wandb.project=MuonQ \
    +logging_params.wandb.name=${RUN_NAME} \
  |& tee logs/${RUN_NAME}.log
