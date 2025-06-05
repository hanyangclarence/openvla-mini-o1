#!/bin/bash

echo "Starting training..."

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=lo
# export NCCL_DEBUG=INFO
# export NCCL_NVLS_ENABLE=1
# export NCCL_P2P_LEVEL=NVL
# export OMP_NUM_THREADS=4
# export NCCL_TIMEOUT=1200


export HF_TOKEN=
export WANDB_API_KEY=28b3c634497c0dc6c16767729d4719b1012a94f2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
#   --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-roboverse" \
#   --data_root_dir data/1.0.0 \
#   --run_root_dir logs/ \
#   --wandb_project "openvla" \
#   --wandb_entity "szang" \
#   --expected_world_size 8 \
#   --global_batch_size 256 \
#   --per_device_batch_size 32 \
#   --pretrained_checkpoint \
#   logs/prism-qwen25-dinosiglip-224px+0_5b+mx-roboverse+n1+b32+x7/checkpoints/step-007500-epoch-00-loss=1.4169.pt\
#   --resume_step 7500 \
#   --resume_epoch 0 \
#   --is_resume true 

# warmup
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir dataset/rl_bench_o1_dataset/3.0.0 \
  --dataset_name rlbencho1 \
  --run_root_dir logs \
  --lora_rank 32 \
  --batch_size 5 \
  --num_images_in_input 2 \
  --use_proprio True \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project "embodied_o1" \
  --wandb_entity "mahlerrrr76" \
  --save_steps 20 \
  --validation_steps 10 \
  --generate_steps 20 \
  --debug False \
  --run_id_note warmup \
  --save_latest_checkpoint_only False

  # torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  # --vla_path "openvla/openvla-7b" \
  # --data_root_dir dataset/rl_bench_o1_dataset/2.0.0 \
  # --dataset_name rlbencho1 \
  # --run_root_dir logs \
  # --lora_rank 32 \
  # --batch_size 5 \
  # --num_images_in_input 2 \
  # --use_proprio True \
  # --grad_accumulation_steps 1 \
  # --learning_rate 5e-4 \
  # --image_aug False \
  # --wandb_project "embodied_o1" \
  # --wandb_entity "mahlerrrr76" \
  # --save_steps 500 \
  # --validation_steps 200 \
  # --generate_steps 50 \
  # --debug False
