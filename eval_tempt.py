from transformers import AutoProcessor, AutoModelForVision2Seq
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
from prismatic.vla.action_tokenizer import ActionTokenizer


def _process_pose_to_state(pose_dict):
    # Extract position (3D) and orientation (4D quaternion)
    pos = np.array(pose_dict['pos'], dtype=np.float32)  # [x, y, z]
    ori = np.array(pose_dict['ori'], dtype=np.float32)  # [qx, qy, qz, qw]
    
    # Convert gripper state to float (1D)
    gripper = 0.0 if pose_dict['gripper_open'] else 1.0
    
    # Concatenate into 8D vector
    state = np.concatenate([pos, ori, [gripper]], dtype=np.float32)
    return state[None, ...]  # (1, 8)


def invert_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    return 1 - actions


saved_model_directory = "openvla/openvla-7b"
dataset_metadata_path = "logs/openvla-7b+rlbencho1+b10+lr-0.0005+lora-r32+dropout-0.0/dataset_statistics.json"
lora_directory = "lora_logs/openvla-7b+rlbencho1+b10+lr-0.0005+lora-r32+dropout-0.0"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
processor = AutoProcessor.from_pretrained(
    saved_model_directory,
    trust_remote_code=True
)
base_model = AutoModelForVision2Seq.from_pretrained(
    saved_model_directory,
    torch_dtype=torch.bfloat16,    # Or your preferred dtype for inference (e.g., torch.float32)
    low_cpu_mem_usage=True,        # Optional: helps with memory if loading large models
    trust_remote_code=True,
).to(device)
model = PeftModel.from_pretrained(base_model, lora_directory)
model = model.merge_and_unload()
model.eval()

# Create Action Tokenizer
action_tokenizer = ActionTokenizer(processor.tokenizer)

dataset_metadata = json.load(open(dataset_metadata_path, 'r'))

correct_action_token_count = 0
total_action_token_count = 0
incorrect_format_count = 0
l1_dist_list = []
all_transitions = glob.glob("/gpfs/yanghan/data/runs_vla_data/val/*/0/video/*")
for idx, path in enumerate(all_transitions):
    if "expert" in path:
        obs_path = f"{path}/front_rgb/begin.png"
    elif "perturb" in path:
        obs_path = f"{path}/front_rgb/end.png"
    else:
        continue
    json_path = f"{path}/info.json"
    if not os.path.exists(obs_path) or not os.path.exists(json_path):
        continue
    
    json_data = json.load(open(json_path, 'r'))
    task_instruction = json_data['lang_goal']
    
    image = Image.open(obs_path).convert("RGB")
    prompt = f"In: What action should the robot take to {task_instruction}?\nOut:"
    
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    generated_ids = model.generate(**inputs, max_new_tokens=200)[0]
    pred_action_ids = generated_ids[generated_ids > action_tokenizer.action_token_begin_idx]
    
    output_str = processor.tokenizer.decode(
        generated_ids.cpu().numpy().tolist(),
        skip_special_tokens=False
    )
    
    if "expert" in path:
        curr_pose = _process_pose_to_state(json_data['prev_pose'])
        next_pose = _process_pose_to_state(json_data['current_pose'])
    else:
        assert "perturb" in path
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
    low = dataset_metadata['action']['q01']
    high = dataset_metadata['action']['q99']
    mask = dataset_metadata['action']['mask']
    
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
    
    if len(pred_action_ids) == len(gt_action_ids):
        correct_action_token_count += np.sum(pred_action_ids == gt_action_ids)
    else:
        incorrect_format_count += 1
    total_action_token_count += len(gt_action_ids)
    
    if len(pred_action_ids) == len(gt_action_ids):
        pred_action = action_tokenizer.decode_token_ids_to_actions(pred_action_ids)
        action_l1_distance = np.mean(np.abs(pred_action - gt_action))
        l1_dist_list.append(action_l1_distance)

    action_token_accuracy = correct_action_token_count / total_action_token_count
    print(f"{idx + 1}/{len(all_transitions)}: Action Token Accuracy: {action_token_accuracy:.4f}, "
          f"Incorrect Format Count: {incorrect_format_count}, "
          f"Average L1 Distance: {np.mean(l1_dist_list) if l1_dist_list else 0:.4f}")
