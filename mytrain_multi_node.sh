#!/bin/bash

echo "Starting training..."

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=$(ip -o -4 route show to default | awk '{print $5}' | head -n 1)
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=1
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=1200


export HF_TOKEN=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
export WANDB_API_KEY=f02cb01552baf522a9cf61ea54648decbdb3c7e9
export HF_HOME=cache
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes=2 --nproc-per-node=8 \
  --node-rank=1 --master-addr=n0261.savio3 --master-port=29500 \
  vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-roboverse" \
  --data_root_dir data/1.0.0 \
  --run_root_dir logs/ \
  --wandb_project "openvla" \
  --wandb_entity "szang" \
  --expected_world_size 16 \
  --global_batch_size 512 \
  --per_device_batch_size 32 \
  --pretrained_checkpoint \
    logs/prism-qwen25-dinosiglip-224px+0_5b+mx-roboverse+n2+b32+x7/checkpoints/step-007500-epoch-00-loss=1.2921.pt\
  --resume_step 7500 \
  --resume_epoch 0  \
  --is_resume true
