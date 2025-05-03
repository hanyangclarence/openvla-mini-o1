#!/bin/bash
#SBATCH --job-name=openvla              # Job name
#SBATCH --output=/global/home/users/ghr/Projects/slurm_logs/output_%j.log         # Standard output and error log (%j expands to jobID)
#SBATCH --error=/global/home/users/ghr/Projects/slurm_logs/error_%j.log           # Separate error log file
#SBATCH --ntasks=1                     # Number of tasks (1 task = 1 CPU)
#SBATCH --cpus-per-task=16             # Number of CPU cores per task
#SBATCH --mem=1500G                       # Total memory for the job
#SBATCH --time=144:00:00                # Time limit (hh:mm:ss)
#SBATCH --partition=bair            # Partition (queue) to submit to
#SBATCH --mail-type=END,FAIL           # Send email at job end or failure
#SBATCH --mail-user=1316294604@qq.com  # Email to send notifications to
#SBATCH --gres=gpu:A100:8                   # Request 1 GPU (optional)
#SBATCH --account=bair
#SBATCH --qos=bair_gpu

eval "$(/global/home/users/ghr/anaconda3/bin/conda shell.bash hook)"

conda activate openvla_szang

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1 
export NCCL_SOCKET_IFNAME=lo                 
export NCCL_NVLS_ENABLE=1                  
export NCCL_P2P_LEVEL=NVL                  
export NCCL_TIMEOUT=1200                   
export OMP_NUM_THREADS=4                   
export HF_TOKEN=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  
export WANDB_API_KEY=f02cb01552baf522a9cf61ea54648decbdb3c7e9                 
export HF_HOME=cache                       
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --nproc_per_node=8 --nnodes=1 vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-roboverse" \
  --data_root_dir data/1.0.0 \
  --run_root_dir logs/ \
  --wandb_project "openvla" \
  --wandb_entity "szang" \
  --expected_world_size 8 \
  --global_batch_size  256 \
  --per_device_batch_size 32 \
