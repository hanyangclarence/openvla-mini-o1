#!/bin/bash

echo "Starting training..."

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=1
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=1200


export HF_TOKEN=
export WANDB_API_KEY=f02cb01552baf522a9cf61ea54648decbdb3c7e9
export HF_HOME=cache
export CUDA_VISIBLE_DEVICES=4,5

torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-roboverse" \
  --run_id "prism-qwen25-dinosiglip-224px+0_5b+mx-roboverse+n0+b32+x7-pickcubel0" \
  --data_root_dir pickcubel0/1.0.0 \
  --run_root_dir logs/ \
  --wandb_project "openvla" \
  --wandb_entity "szang" \
  --expected_world_size 2 \
  --global_batch_size 64 \
  --per_device_batch_size 32 \
