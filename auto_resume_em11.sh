#!/bin/bash

CHECKPOINT_DIR="logs/prism-qwen25-dinosiglip-224px+0_5b+mx-roboverse+n0+b16+x7/checkpoints"
MEMORY_THRESHOLD=20  # GB

# Function to set common environment variables
set_environment() {
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
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
}

# Function to get latest checkpoint info
get_latest_checkpoint() {
    LATEST_CHECKPOINT=$(ls -1 ${CHECKPOINT_DIR}/step-*.pt | sort -V | tail -n 1)
    if [ -z "${LATEST_CHECKPOINT}" ]; then
        echo "No checkpoint found in ${CHECKPOINT_DIR}"
        return 1
    fi
    
    # Extract numbers using more precise patterns
    STEP_NUM=$(echo ${LATEST_CHECKPOINT} | grep -o 'step-[0-9]\+' | cut -d'-' -f2)
    EPOCH_NUM=$(echo ${LATEST_CHECKPOINT} | grep -o 'epoch-[0-9]\+' | cut -d'-' -f2)
    
    # Verify we got valid integers
    if ! [[ "$STEP_NUM" =~ ^[0-9]+$ ]] || ! [[ "$EPOCH_NUM" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Failed to extract valid integer values from checkpoint filename"
        echo "Checkpoint: ${LATEST_CHECKPOINT}"
        echo "Extracted STEP_NUM: ${STEP_NUM}"
        echo "Extracted EPOCH_NUM: ${EPOCH_NUM}"
        return 1
    fi

    # Convert to integers explicitly
    STEP_NUM=$((10#${STEP_NUM}))  # Force base-10 interpretation
    EPOCH_NUM=$((10#${EPOCH_NUM}))

    echo "Found checkpoint: ${LATEST_CHECKPOINT}"
    echo "Step: ${STEP_NUM} (integer)"
    echo "Epoch: ${EPOCH_NUM} (integer)"
    return 0
}

# Function to terminate training process and wait for cleanup
terminate_training() {
    local pid=$1
    echo "Attempting to terminate training process (PID: $pid)..."
    
    # Get all child PIDs before terminating parent
    local children=$(pgrep -P $pid)
    echo "Found child processes: $children"
    
    # Try graceful termination first
    kill -TERM $pid 2>/dev/null
    
    # Wait up to 60 seconds for graceful shutdown
    local count=0
    while kill -0 $pid 2>/dev/null && [ $count -lt 60 ]; do
        echo "Waiting for process to terminate... ($count/60s)"
        sleep 1
        count=$((count + 1))
    done
    
    # If still running, force kill parent and children
    if kill -0 $pid 2>/dev/null; then
        echo "Process still running, force killing..."
        kill -9 $pid 2>/dev/null
        [ ! -z "$children" ] && kill -9 $children 2>/dev/null
    fi
    
    # Additional wait to ensure GPU memory is freed
    echo "Waiting for system cleanup..."
    sleep 10  # 10 seconds for thorough cleanup
    
    # Verify no related processes remain
    if pgrep -P $pid >/dev/null; then
        echo "ERROR: Some child processes still exist"
        return 1
    fi
    
    echo "Training process terminated successfully"
    return 0
}

# Function to run training with memory monitoring and exit handling
run_training_with_monitoring() {
    local IS_RESUME=$1
    set_environment

    # Prepare base command
    local BASE_CMD="torchrun --standalone --nnodes 1 --nproc-per-node 6 vla-scripts/train.py \
        --vla.type 'prism-qwen25-dinosiglip-224px+0_5b+mx-roboverse' \
        --data_root_dir data/1.0.0 \
        --run_root_dir logs/ \
        --wandb_project 'openvla' \
        --wandb_entity 'szang' \
        --expected_world_size 6 \
        --global_batch_size 96 \
        --per_device_batch_size 16" 

    # Add resume-specific parameters if resuming
    if [ "$IS_RESUME" = true ]; then
        echo "Starting training in resume mode..."
        eval "$BASE_CMD \
            --pretrained_checkpoint '${LATEST_CHECKPOINT}' \
            --resume_step '${STEP_NUM}' \
            --resume_epoch '${EPOCH_NUM}' \
            --is_resume true" &
    else
        echo "Starting fresh training..."
        eval "$BASE_CMD" &
    fi

    # Get the actual torchrun process PID
    sleep 2  # Give torchrun a moment to start
    TRAIN_PID=$(pgrep -f "torchrun.*train.py")
    
    if [ -z "$TRAIN_PID" ]; then
        echo "ERROR: Failed to find torchrun process"
        exit 1
    fi
    echo "Found torchrun process with PID: $TRAIN_PID"

    # Monitor process and memory
    while true; do
        # Check if process is still running
        if ! kill -0 $TRAIN_PID 2>/dev/null; then
            echo "WARNING: Training process exited unexpectedly"
            sleep 60  # Wait for potential checkpoint to be saved
            
            if get_latest_checkpoint; then
                echo "Found checkpoint after exit - attempting to resume..."
                run_training_with_monitoring true
                return
            else
                echo "ERROR: No checkpoint found after unexpected exit"
                exit 1
            fi
        fi

        # Check memory usage
        AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
        if [ ${AVAILABLE_MEM} -lt ${MEMORY_THRESHOLD} ]; then
            echo "WARNING: Available memory (${AVAILABLE_MEM}GB) is below threshold (${MEMORY_THRESHOLD}GB)"
            
            # Terminate and wait for cleanup
            if ! terminate_training $TRAIN_PID; then
                echo "ERROR: Failed to terminate training properly"
                exit 1
            fi
            
            # Try to resume from new checkpoint
            if get_latest_checkpoint; then
                echo "Attempting to resume from new checkpoint..."
                run_training_with_monitoring true
                return
            else
                echo "ERROR: No valid checkpoint found for resume"
                exit 1
            fi
        fi
        
        sleep 60  # Check every minute
    done

    # We shouldn't reach here, but just in case
    wait $TRAIN_PID
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Training process exited with code $EXIT_CODE"
        # Try to resume if there's a checkpoint
        if get_latest_checkpoint; then
            echo "Found checkpoint after exit - attempting to resume..."
            run_training_with_monitoring true
            return
        fi
        exit $EXIT_CODE
    fi
}

# Main execution flow
echo "Checking for existing checkpoints..."
if get_latest_checkpoint; then
    echo "Found existing checkpoint - attempting to resume..."
    run_training_with_monitoring true
else
    echo "No existing checkpoints found - starting fresh training..."
    run_training_with_monitoring false
fi