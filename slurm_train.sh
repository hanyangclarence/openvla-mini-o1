#!/bin/bash
#SBATCH --job-name=openvla              # Job name
#SBATCH --output=/global/home/users/ghr/Projects/slurm_logs/output_%j.log         # Standard output and error log (%j expands to jobID)
#SBATCH --error=/global/home/users/ghr/Projects/slurm_logs/error_%j.log           # Separate error log file
#SBATCH --nodes=4                       # 🔧 Request 2 nodes
#SBATCH --ntasks-per-node=1             # 🔧 1 task per node
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
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
export NCCL_IB_DISABLE=0 
export NCCL_SOCKET_IFNAME=$(ip -o -4 route show to default | awk '{print $5}' | head -n 1)                 
export NCCL_NVLS_ENABLE=1                  
export NCCL_P2P_LEVEL=NVL                  
export NCCL_TIMEOUT=1200                   
export OMP_NUM_THREADS=4                   
export HF_TOKEN=  
export WANDB_API_KEY=f02cb01552baf522a9cf61ea54648decbdb3c7e9                 
export HF_HOME=cache                       
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "Allocated nodes:"
echo "$SLURM_JOB_NODELIST"

echo "Resolved hostnames:"
scontrol show hostnames $SLURM_JOB_NODELIST

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${nodes[0]}
MASTER_PORT=29500
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID
GPUS_PER_NODE=8

NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
MASTER_NODE=$(head -n 1 <<< "$NODELIST")
NODE_COUNT=0
NODE_NUM=($(echo $NODELIST | tr " " "\n" | wc -l))

for NODE in $NODELIST; do
    if [ "$NODE" == "$MASTER_NODE" ]; then
        srun --nodes=1 --ntasks=1 -w $NODE torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NODE_NUM --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT vla-scripts/train.py --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-roboverse" --data_root_dir data/1.0.0 --run_root_dir logs/ --wandb_project "openvla" --wandb_entity "szang" --expected_world_size 32 --global_batch_size 1024 --per_device_batch_size 32 &
    else
        ((NODE_COUNT++))
        srun --nodes=1 --ntasks=1 -w $NODE torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NODE_NUM --node_rank=$NODE_COUNT  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT vla-scripts/train.py --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-roboverse" --data_root_dir data/1.0.0 --run_root_dir logs/ --wandb_project "openvla" --wandb_entity "szang" --expected_world_size 32 --global_batch_size 1024 --per_device_batch_size 32 &
    fi
done
wait
