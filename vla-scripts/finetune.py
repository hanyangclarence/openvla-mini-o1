"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
import wandb
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.training.finetune_utils import *

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
IGNORE_INDEX = -100

# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Architecture
    num_images_in_input: int = 1                                   # Number of images in the VLA input (default: 1)
    use_proprio: bool = False
    
    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    validation_steps: int = 1000                                    # Interval for validation
    generate_steps: int = 50
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    lr_warmup_steps: int = 0                                        # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000                           # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    debug: bool = False


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    if cfg.debug:
        import pdb
        pdb.set_trace()
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = OpenVLAForActionPrediction.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": 8},
        )
    
    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    
    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )
    
    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    
    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1
    
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    val_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=False,
        train=False,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_reasoning_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                    proprio=batch["proprio"] if cfg.use_proprio else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            pred_ids = output.logits[:, NUM_PATCHES : -1].argmax(dim=2)
            gt_ids = batch["labels"][:, 1:].to(device_id)
            action_mask = gt_ids > action_tokenizer.action_token_begin_idx
            reasoning_mask = (gt_ids != IGNORE_INDEX) & torch.logical_not(action_mask)

            # Compute Accuracy
            action_accuracy = compute_token_accuracy(pred_ids, gt_ids, action_mask)
            reasoning_accuracy = compute_token_accuracy(pred_ids, gt_ids, reasoning_mask)
            action_l1_loss = compute_actions_l1_loss(
                action_tokenizer, pred_ids, gt_ids, action_mask
            )

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_reasoning_accuracies.append(reasoning_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_reasoning_accuracy = sum(recent_reasoning_accuracies) / len(recent_reasoning_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Push Metrics to W&B (every 10 gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "Train/train_loss": smoothened_loss,
                        "Train/action_accuracy": smoothened_action_accuracy,
                        "Train/reasoning_accuracy": smoothened_reasoning_accuracy,
                        "Train/l1_loss": smoothened_l1_loss,
                    },
                    step=log_step,
                )
            
            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "Train/learning rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            # Test model on validation set
            if gradient_step_idx > 0 and gradient_step_idx % cfg.validation_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Running validation for Step {gradient_step_idx}...")
                
                vla.eval()
                val_losses = []
                val_action_accuracies = []
                val_reasoning_accuracies = []
                val_l1_losses = []
                
                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(dataloader_val):
                        # Limit validation to a fixed number of batches to speed it up, e.g., 50 batches
                        # Adjust this number based on your validation set size and desired frequency
                        if val_batch_idx >= 50: 
                            break
                        
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            val_output: CausalLMOutputWithPast = vla(
                                input_ids=val_batch["input_ids"].to(device_id),
                                attention_mask=val_batch["attention_mask"].to(device_id),
                                pixel_values=val_batch["pixel_values"].to(torch.bfloat16).to(device_id),
                                labels=val_batch["labels"],
                                proprio=batch["proprio"] if cfg.use_proprio else None,
                                proprio_projector=proprio_projector if cfg.use_proprio else None,
                            )
                            val_loss = val_output.loss

                        val_losses.append(val_loss.item())

                        # Compute Accuracy and L1 Loss for Logging
                        val_action_logits = val_output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                        val_action_preds = val_action_logits.argmax(dim=2)
                        val_action_gt = val_batch["labels"][:, 1:].to(val_action_preds.device)
                        val_mask = val_action_gt > action_tokenizer.action_token_begin_idx
                        val_reasoning_mask = (val_action_gt != IGNORE_INDEX) & torch.logical_not(val_mask)

                        if val_mask.sum().item() > 0:
                            val_correct_preds = (val_action_preds == val_action_gt) & val_mask
                            val_action_accuracy = val_correct_preds.sum().float() / val_mask.sum().float()
                            val_action_accuracies.append(val_action_accuracy.item())
                        
                            # Compute L1 Loss on Predicted (Continuous) Actions
                            # Ensure tensors are on CPU for numpy conversion if not already
                            continuous_actions_pred_val = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(val_action_preds[val_mask].cpu().numpy())
                            )
                            continuous_actions_gt_val = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(val_action_gt[val_mask].cpu().numpy())
                            )
                            val_action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred_val, continuous_actions_gt_val)
                            val_l1_losses.append(val_action_l1_loss.item())

                        if val_reasoning_mask.sum().item() > 0:
                            val_correct_reasoning_preds = (val_action_preds == val_action_gt) & val_reasoning_mask
                            val_reasoning_accuracy = val_correct_reasoning_preds.sum().float() / val_reasoning_mask.sum().float()
                            val_reasoning_accuracies.append(val_reasoning_accuracy.item())
                
                # Aggregate metrics from all processes
                # Summing up local sums and then dividing by total count is more robust
                # For simplicity here, we average local averages if number of val batches is fixed and same for all
                
                # Sum of losses from all processes
                total_val_loss_tensor = torch.tensor(sum(val_losses) if val_losses else 0.0, device=device_id)
                dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
                
                # Count of batches processed for loss (can be different if some processes had fewer batches)
                num_val_loss_batches_tensor = torch.tensor(len(val_losses), device=device_id)
                dist.all_reduce(num_val_loss_batches_tensor, op=dist.ReduceOp.SUM)

                avg_val_loss = (total_val_loss_tensor / num_val_loss_batches_tensor).item() if num_val_loss_batches_tensor.item() > 0 else 0

                # Similar aggregation for other metrics
                total_val_action_accuracy_tensor = torch.tensor(sum(val_action_accuracies) if val_action_accuracies else 0.0, device=device_id)
                num_val_action_accuracy_batches_tensor = torch.tensor(len(val_action_accuracies), device=device_id)
                dist.all_reduce(total_val_action_accuracy_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_val_action_accuracy_batches_tensor, op=dist.ReduceOp.SUM)
                avg_val_action_accuracy = (total_val_action_accuracy_tensor / num_val_action_accuracy_batches_tensor).item() if num_val_action_accuracy_batches_tensor.item() > 0 else 0
                
                total_val_reasoning_accuracy_tensor = torch.tensor(sum(val_reasoning_accuracies) if val_reasoning_accuracies else 0.0, device=device_id)
                num_val_reasoning_accuracy_batches_tensor = torch.tensor(len(val_reasoning_accuracies), device=device_id)
                dist.all_reduce(total_val_reasoning_accuracy_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_val_reasoning_accuracy_batches_tensor, op=dist.ReduceOp.SUM)
                avg_val_reasoning_accuracy = (total_val_reasoning_accuracy_tensor / num_val_reasoning_accuracy_batches_tensor).item() if num_val_reasoning_accuracy_batches_tensor.item() > 0 else 0

                total_val_l1_loss_tensor = torch.tensor(sum(val_l1_losses) if val_l1_losses else 0.0, device=device_id)
                num_val_l1_loss_batches_tensor = torch.tensor(len(val_l1_losses), device=device_id)
                dist.all_reduce(total_val_l1_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_val_l1_loss_batches_tensor, op=dist.ReduceOp.SUM)
                avg_val_l1_loss = (total_val_l1_loss_tensor / num_val_l1_loss_batches_tensor).item() if num_val_l1_loss_batches_tensor.item() > 0 else 0

                if distributed_state.is_main_process:
                    wandb.log(
                        {
                            "Val/loss": avg_val_loss,
                            "Val/action_accuracy": avg_val_action_accuracy,
                            "Val/reasoning_accuracy": avg_val_reasoning_accuracy,
                            "Val/l1_loss": avg_val_l1_loss,
                        },
                        step=gradient_step_idx,
                    )
                    print(f"Step {gradient_step_idx}: Val Loss: {avg_val_loss:.4f}, Val Action Acc: {avg_val_action_accuracy:.4f}")

                vla.train() # Switch back to training mode
                dist.barrier() # Ensure all processes finish validation before continuing training

            # Generate outputs as validation
            if gradient_step_idx > 0 and gradient_step_idx % cfg.generate_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Generating outputs for Step {gradient_step_idx}...")
                
                # generate sample for each item in the batch
                vla.eval()
                correct_token_count = 0
                correct_trans_count = 0
                correct_rot_count = 0
                correct_gripper_count = 0
                total_sample_count = 0
                with torch.no_grad():
                    for i in range(batch["input_ids"].shape[0]):
                        input_ids_sample = batch["input_ids"][i]
                        labels_sample = batch["labels"][i]
                        attention_mask_sample = batch["attention_mask"][i]
                        pixel_values_sample = batch["pixel_values"][i:i + 1].to(vla.module.dtype).to(device_id)
                        
                        # Determine prompt length
                        first_target_indices = (labels_sample != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                        prompt_len = first_target_indices[0].item()
                        
                        input_ids_sample = input_ids_sample[:prompt_len].unsqueeze(0).to(device_id)
                        attention_mask_sample = attention_mask_sample[:prompt_len].unsqueeze(0).to(device_id)
                        
                        generated_ids = vla.module.generate(
                            input_ids=input_ids_sample,
                            attention_mask=attention_mask_sample,
                            pixel_values=pixel_values_sample,
                            proprio=batch["proprio"][i:i + 1].to(device_id) if cfg.use_proprio else None,
                            proprio_projector=proprio_projector if cfg.use_proprio else None,
                            eos_token_id=processor.tokenizer.eos_token_id,
                            pad_token_id=processor.tokenizer.pad_token_id,
                            max_new_tokens=200,
                        )[0]
                        
                        # calculate action token accuracy
                        gt_action_ids = labels_sample[labels_sample > action_tokenizer.action_token_begin_idx]
                        pred_action_ids = generated_ids[generated_ids > action_tokenizer.action_token_begin_idx].cpu()
                        
                        total_sample_count += 1
                        if len(gt_action_ids) != len(pred_action_ids):
                            continue
                        correct_token_count += (gt_action_ids == pred_action_ids).sum().item()
                        correct_trans_count += (gt_action_ids[:3] == pred_action_ids[:3]).sum().item()
                        correct_rot_count += (gt_action_ids[3:6] == pred_action_ids[3:6]).sum().item()
                        correct_gripper_count += (gt_action_ids[6] == pred_action_ids[6]).sum().item()
                
                # aggregate metrics from all processes
                correct_token_count_tensor = torch.tensor(correct_token_count, device=device_id)
                correct_trans_count_tensor = torch.tensor(correct_trans_count, device=device_id)
                correct_rot_count_tensor = torch.tensor(correct_rot_count, device=device_id)
                correct_gripper_count_tensor = torch.tensor(correct_gripper_count, device=device_id)
                total_sample_count_tensor = torch.tensor(total_sample_count, device=device_id)
                dist.all_reduce(correct_token_count_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(correct_trans_count_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(correct_rot_count_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(correct_gripper_count_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_sample_count_tensor, op=dist.ReduceOp.SUM)
                
                if distributed_state.is_main_process:
                    action_token_accuracy = correct_token_count_tensor.item() / (total_sample_count_tensor.item() * 7) if total_sample_count_tensor.item() > 0 else 0
                    trans_token_accuracy = correct_trans_count_tensor.item() / (total_sample_count_tensor.item() * 3) if total_sample_count_tensor.item() > 0 else 0
                    rot_token_accuracy = correct_rot_count_tensor.item() / (total_sample_count_tensor.item() * 3) if total_sample_count_tensor.item() > 0 else 0
                    gripper_token_accuracy = correct_gripper_count_tensor.item() / total_sample_count_tensor.item() if total_sample_count_tensor.item() > 0 else 0
                    wandb.log(
                        {
                            "Gen/action_token_accuracy": action_token_accuracy,
                            "Gen/trans_token_accuracy": trans_token_accuracy,
                            "Gen/rot_token_accuracy": rot_token_accuracy,
                            "Gen/gripper_token_accuracy": gripper_token_accuracy,
                        },
                        step=gradient_step_idx,
                    )
                    print(
                        f"Step {gradient_step_idx}: Action Token Accuracy: {action_token_accuracy:.4f}, {trans_token_accuracy:.4f}, "
                        f"{rot_token_accuracy:.4f}, {gripper_token_accuracy:.4f}")
                vla.train()
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
