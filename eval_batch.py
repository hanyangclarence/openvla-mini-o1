from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
import torch
from PIL import Image
from peft import LoraConfig, PeftModel
from typing import Optional
import glob
import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt
from dataclasses import dataclass
import random

from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.eval_utils import get_models, get_proprio_projector, _process_pose_to_state, invert_gripper_actions, prepare_inputs

import pdb
pdb.set_trace()

@dataclass
class GenerateConfig:
    pretrained_checkpoint: str = None,
    num_images_in_input: int = 1,
    use_proprio: bool = True,
    unnorm_key: str = None,
    center_crop: bool = True,
    device: torch.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# Init models
cfg = GenerateConfig(
    pretrained_checkpoint="/gpfs/yanghan/openvla-mini-o1/logs/openvla-7b+rlbencho1+b5+lr-0.0005+lora-r32+dropout-0.0--image_aug--2000_chkpt",
    num_images_in_input=2,
    use_proprio=True,
    unnorm_key="rlbencho1",
    center_crop=True,
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
)
model, processor = get_models(cfg)
proprio_projector = get_proprio_projector(
    cfg, llm_dim=model.llm_dim, proprio_dim=8
)

# Create Action Tokenizer
action_tokenizer = ActionTokenizer(processor.tokenizer)

dataset_metadata = model.norm_stats['rlbencho1']

correct_action_token_count = 0
correct_transition_token_count = 0
correct_rotation_token_count = 0
correct_gripper_token_count = 0
correct_format_count = 0
incorrect_format_count = 0
l1_dist_list = []
transition_l1_dist_list = []
rotation_l1_dist_list = []
gripper_l1_dist_list = []
BATCH_SIZE = 4
all_transitions = glob.glob("/gpfs/yanghan/data/runs_vla_data/val/*/0/video/*")
all_inputs = []
all_jsons = []
random.shuffle(all_transitions)
for idx, path in enumerate(all_transitions):
    if "expert" in path:
        obs_path = f"{path}/front_rgb/begin.png"
        wrist_obs_path = f"{path}/wrist_rgb/begin.png"
    elif "perturb" in path:
        obs_path = f"{path}/front_rgb/end.png"
        wrist_obs_path = f"{path}/wrist_rgb/end.png"
    else:
        continue
    json_path = f"{path}/info.json"
    if not os.path.exists(obs_path) or not os.path.exists(json_path):
        continue
    
    json_data = json.load(open(json_path, 'r'))
    task_instruction = json_data['lang_goal']
    
    image = Image.open(obs_path).convert("RGB")
    wrist_image = Image.open(wrist_obs_path).convert("RGB")
    if "expert" in path:
        state = _process_pose_to_state(json_data['prev_pose'])
    else:
        state = _process_pose_to_state(json_data['current_pose'])
    
    observation = {
        "full_image": np.array(image).astype(np.uint8),
        "wrist_image": np.array(wrist_image).astype(np.uint8),
        "state": state,
        "task_description": task_instruction,
    }
    all_inputs.append(
        prepare_inputs(
            cfg=cfg, vla=model, processor=processor, obs=observation, task_label=task_instruction,
        )
    )
    all_jsons.append(json_data)
    
    if len(all_inputs) < BATCH_SIZE:
        continue
    
    # collate inputs
    all_input_ids = [x['input_ids'] for x in all_inputs]
    all_attention_mask = [x['attention_mask'] for x in all_inputs]
    max_input_length = max(x.shape[-1] for x in all_input_ids)
    padded_input_ids = torch.full(
        (BATCH_SIZE, max_input_length),
        fill_value=processor.tokenizer.pad_token_id,
        dtype=torch.long,
        device=cfg.device
    )
    padded_attention_mask = torch.zeros(
        (BATCH_SIZE, max_input_length),
        dtype=torch.long,
        device=cfg.device
    )
    for i in range(BATCH_SIZE):
        padded_input_ids[i, :all_input_ids[i].shape[-1]] = all_input_ids[i]
        padded_attention_mask[i, :all_attention_mask[i].shape[-1]] = all_attention_mask[i]
    
    all_pixel_values = torch.cat([x['pixel_values'] for x in all_inputs], dim=0)
    all_proprio = torch.cat([x['proprio'] for x in all_inputs], dim=0)
    
    # batch generation
    all_generated_ids = model.generate(
        input_ids=padded_input_ids,
        attention_mask=padded_attention_mask,
        pixel_values=all_pixel_values,
        proprio=all_proprio,
        proprio_projector=proprio_projector,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        max_new_tokens=200,
    )
    
    for i in range(BATCH_SIZE):
        json_data = all_jsons[i]
        generated_ids = all_generated_ids[i]
    
        pred_action_ids = generated_ids[(generated_ids > action_tokenizer.action_token_begin_idx) & (generated_ids != processor.tokenizer.pad_token_id)].cpu().numpy()
        
        output_str = processor.tokenizer.decode(
            generated_ids.cpu().numpy().tolist(),
            skip_special_tokens=False
        )
        
        is_perturb = "correct_pose" in json_data
        if not is_perturb:
            curr_pose = _process_pose_to_state(json_data['prev_pose'])
            next_pose = _process_pose_to_state(json_data['current_pose'])
        else:
            curr_pose = _process_pose_to_state(json_data['current_pose'])
            next_pose = _process_pose_to_state(json_data['correct_pose'])
        
        eef_position_proprio, eef_orientation_proprio, gripper_proprio = tf.split(curr_pose, [3,4,1], axis=1)  # (T,3) (T,4) (T,1)
        eef_position_control, eef_orientation_control, gripper_control = tf.split(next_pose, [3,4,1], axis=1)  # (T,3) (T,4) (T,1)
        
        action_gripper = invert_gripper_actions(gripper_control) # +1 = open, 0 = close
        
        action_delta_xyz = eef_position_control - eef_position_proprio # (T, 3)
        
        # quaternions in rlbench and tfgraphics are all in format xyzw, so we don't need further conversion
        delta_eef_orientation_proprio = tfgt.quaternion.multiply(
            eef_orientation_control, tfgt.quaternion.inverse(eef_orientation_proprio)
        )
        delta_eef_orientation_proprio = tfgt.quaternion.normalize(delta_eef_orientation_proprio)
        action_delta_rpy = tfgt.euler.from_quaternion(delta_eef_orientation_proprio)
        
        # resolve NaN values in action_delta_rpy
        action_delta_rpy = tf.where(tf.math.is_nan(action_delta_rpy), tf.zeros_like(action_delta_rpy), action_delta_rpy)
        
        gt_action = tf.concat([action_delta_xyz, action_delta_rpy, action_gripper], axis=-1)
        
        # normalize action with metadata, with default q99 method
        low = tf.constant(dataset_metadata['action']['q01'], dtype=tf.float32)
        high = tf.constant(dataset_metadata['action']['q99'], dtype=tf.float32)
        mask = tf.constant(dataset_metadata['action']['mask'], dtype=tf.bool)
        
        gt_action = tf.where(
            mask,
            tf.clip_by_value(2 * (gt_action - low) / (high - low + 1e-8) - 1, -1, 1),
            gt_action
        )
        
        gt_action = np.array(gt_action, dtype=np.float32)[0]
        
        # Tokenize the action into ids
        gt_action_ids = np.digitize(
            np.clip(gt_action, a_min=action_tokenizer.min_action, a_max=action_tokenizer.max_action),
            action_tokenizer.bins,
        )
        gt_action_ids = action_tokenizer.tokenizer_len - gt_action_ids
        
        if len(pred_action_ids) == len(gt_action_ids):
            correct_action_token_count += np.sum(pred_action_ids == gt_action_ids)
            correct_transition_token_count += np.sum(pred_action_ids[:3] == gt_action_ids[:3])
            correct_rotation_token_count += np.sum(pred_action_ids[3:6] == gt_action_ids[3:6])
            correct_gripper_token_count += np.sum(pred_action_ids[6] == gt_action_ids[6])
            correct_format_count += 1
        else:
            incorrect_format_count += 1
        
        if len(pred_action_ids) == len(gt_action_ids):
            pred_action = action_tokenizer.decode_token_ids_to_actions(pred_action_ids)
            action_l1_distance = np.mean(np.abs(pred_action - gt_action))
            transition_l1_distance = np.mean(np.abs(pred_action[:3] - gt_action[:3]))
            rotation_l1_distance = np.mean(np.abs(pred_action[3:6] - gt_action[3:6]))
            gripper_l1_distance = np.mean(np.abs(pred_action[6] - gt_action[6]))
            transition_l1_dist_list.append(transition_l1_distance)
            rotation_l1_dist_list.append(rotation_l1_distance)
            gripper_l1_dist_list.append(gripper_l1_distance)
            l1_dist_list.append(action_l1_distance)

        action_accuracy = correct_action_token_count / (correct_format_count * 7)
        transition_accuracy = correct_transition_token_count / (correct_format_count * 3)
        rotation_accuracy = correct_rotation_token_count / (correct_format_count * 3)
        gripper_accuracy = correct_gripper_token_count / (correct_format_count * 1)
        transition_accuracy 
        print(f"{i + idx + 1 - BATCH_SIZE}/{len(all_transitions)}: Action Accuracy: {action_accuracy:.4f}, {transition_accuracy:.4f}, "
            f"{rotation_accuracy:.4f}, {gripper_accuracy:.4f}, "
            f"Incorrect Count: {incorrect_format_count}, "
            f"L1 Distance: {np.mean(l1_dist_list) if l1_dist_list else 0:.4f}, {np.mean(transition_l1_dist_list) if transition_l1_dist_list else 0:.4f}, "
            f"{np.mean(rotation_l1_dist_list) if rotation_l1_dist_list else 0:.4f}, "
                f"{np.mean(gripper_l1_dist_list) if gripper_l1_dist_list else 0:.4f}")

    all_inputs = []
    all_jsons = []
    if idx % BATCH_SIZE == 0:
        torch.cuda.empty_cache()
        