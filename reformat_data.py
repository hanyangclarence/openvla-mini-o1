task_list = [
    "put_item_in_drawer",             # 0                 
    "reach_and_drag",                 # 1                 
    "turn_tap",                       # 2  --> [0:3]      
    "slide_block_to_color_target",    # 3                 
    "open_drawer",                    # 4                 
    "put_groceries_in_cupboard",      # 5  --> [3:6]      
    "place_shape_in_shape_sorter",    # 6                 
    "put_money_in_safe",              # 7                 
    "push_buttons",                   # 8  --> [6:9]      
    "close_jar",                      # 9                 
    "stack_blocks",                   # 10                
    "place_cups",                     # 11 --> [9:12]     
    "place_wine_at_rack_location",    # 12                
    "light_bulb_in",                  # 13                
    "sweep_to_dustpan_of_size",       # 14 --> [12:15]    
    "insert_onto_square_peg",         # 15                
    "meat_off_grill",                 # 16                
    "stack_cups",                     # 17 --> [15:18]
]

import os
from os.path import join as pjoin
import json
import argparse
import pickle
import numpy as np

from prismatic.models import load
from prismatic.vla.action_tokenizer import ACTION_TOKENIZERS


data_dir = "/gpfs/u/scratch/LMCG/LMCGhazh/enroot/rlbench_data/root/RACER-DataGen/racer_datagen/runs_vla_data"
output_dir = "/gpfs/u/scratch/LMCG/LMCGhazh/enroot/rlbench_data/root/RACER-DataGen/racer_datagen/runs_vla_data_lammafact"
obs_window_size = 2
view = "front_rgb"
pose_lower_bound = np.array([-0.31, -0.5, 0.74, -1, -1, -1, -1, 0])
pose_upper_bound = np.array([0.53,   0.5, 1.52, 1, 1, 1, 1, 1])

vlm = load("model/prism-qwen25-extra-dinosiglip-224px-0_5b")
action_tokenizer = ACTION_TOKENIZERS["extra_action_tokenizer"](vlm.llm_backbone.get_tokenizer())


def get_action(curr, next):
    # defined to extract the action
    # currently just use the next action
    return np.concatenate([next.gripper_pose, np.array([next.gripper_open])], axis=0)
    


def action2str(action: np.ndarray) -> str:
    # scale the action to [-1, 1]
    action = (action - pose_lower_bound) / (pose_upper_bound - pose_lower_bound) * 2 - 1
    assert action.min() >= -1 and action.max() <= 1, f"Action out of range: {action.min()} {action.max()}"
    # convert the action to a string of numbers
    action_str = action_tokenizer(action)
    
    ids = action_tokenizer.tokenizer(action_str)["input_ids"]
    action_recon = action_tokenizer.decode_token_ids_to_actions(np.array(ids))
    print(f"Reconstruction loss: {np.abs(action_recon - action).max()}")
    return action_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of the task list")
    parser.add_argument("--end_idx", type=int, default=18, help="End index of the task list")
    parser.add_argument("--max_videos", type=int, default=-1, help="Maximum number of videos to process")
    args = parser.parse_args()


    task_list = task_list[args.start_idx: args.end_idx]  # start_idx is inclusive, end_idx is exclusive
    os.makedirs(output_dir, exist_ok=True)
    img_save_dir = pjoin(output_dir, "images")
    os.makedirs(img_save_dir, exist_ok=True)

    metadata = []
    video_cnt = 0
    for split in ["val", "train"]:
        for task in os.listdir(pjoin(data_dir, split)):
            if args.max_videos > 0 and video_cnt >= args.max_videos:
                break
            
            if task not in task_list:
                continue
            task_dir = pjoin(data_dir, split, task)
            all_episodes = os.listdir(task_dir)
            for episode in all_episodes:
                if args.max_videos > 0 and video_cnt >= args.max_videos:
                    break
                
                all_video_dir = pjoin(task_dir, episode, "video")
                if not os.path.exists(all_video_dir):
                    print(f"Video directory not found: {all_video_dir}")
                    continue
                
                all_keyframes = os.listdir(all_video_dir)
                all_keyframes = [int(f.split("_")[0]) for f in all_keyframes if os.path.isdir(pjoin(all_video_dir, f))]
                all_keyframes = sorted(list(set(all_keyframes)))
                for filename in os.listdir(all_video_dir):
                    file_dir = pjoin(all_video_dir, filename)
                    if not os.path.isdir(file_dir):
                        continue
                    keypose_id = int(filename.split("_")[0])
                    keypose_idx = all_keyframes.index(keypose_id)
                    
                    
                    info_path = pjoin(file_dir, "info.json")
                    if not os.path.exists(info_path):
                        print(f"Info file not found: {info_path}")
                        continue
                    with open(info_path, "r") as f:
                        info = json.load(f)
                        
                    if "expert" in filename:
                        curr_obs_path = pjoin(file_dir, "low_dim_obs_begin.pkl")
                        next_obs_path = pjoin(file_dir, "low_dim_obs_end.pkl")
                        if not os.path.exists(curr_obs_path) or not os.path.exists(next_obs_path):
                            print(f"Observation files not found: {curr_obs_path}, {next_obs_path}")
                            continue
                        curr_obs = pickle.load(open(curr_obs_path, "rb"))
                        next_obs = pickle.load(open(next_obs_path, "rb"))
                        
                        act = get_action(curr=curr_obs, next=next_obs)
                        
                        img_obs_paths = [pjoin(file_dir, view, "begin.png")]
                        if obs_window_size > 1:
                            for i in range(1, obs_window_size):
                                if keypose_idx - i < 0:
                                    break
                                img_obs_paths.append(
                                    pjoin(all_video_dir, f"{all_keyframes[keypose_idx - i]}_expert", view, "begin.png")
                                )
                            img_obs_paths.reverse()
                        if any([not os.path.exists(p) for p in img_obs_paths]):
                            print(f"Image observation files not found: {img_obs_paths}")
                            continue
                        # for each image, save it to the output directory by creating a link
                        new_img_paths = []
                        for img_obs_path in img_obs_paths:
                            img_obs_path = img_obs_path.replace(data_dir, "")
                            img_obs_path = img_obs_path[1:] if img_obs_path.startswith("/") else img_obs_path
                            img_id = "_".join(img_obs_path.split("/"))
                            new_img_path = pjoin(img_save_dir, img_id)
                            if not os.path.exists(new_img_path):
                                os.symlink(pjoin(data_dir, img_obs_path), new_img_path)
                            new_img_paths.append(img_id)
                        
                        in_prompt = ""
                        for _ in range(len(new_img_paths)):
                            in_prompt += "<image>"
                        lang_goal = info["lang_goal"]
                        in_prompt += f"What action should the robot take to {lang_goal}?"
                        
                        if "subgoal" not in info:
                            print(f"Subgoal not found in info: {filename}")
                            continue
                        answer = "<|reason_start|>"
                        subgoal = info["subgoal"]
                        answer += f"Previous action is successful. To achieve the goal, the robot should now {subgoal.lower()}"
                        answer += "<|reason_end|>"
                        # # tempt just write action as a string of numbers
                        # act = act.tolist()
                        # action_encoded = " ".join([str(a) for a in act])
                        # answer += action_encoded
                        answer += action2str(act)
                        
                        metadata.append({
                            "id": video_cnt,
                            "image": new_img_paths[0] if len(new_img_paths) <= 1 else new_img_paths,
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": in_prompt
                                },
                                {
                                    "from": "gpt",
                                    "value": answer
                                }
                            ]
                        })
                    elif "perturb" in filename:
                        curr_obs_path = pjoin(file_dir, "low_dim_obs_end.pkl")
                        next_obs_path = pjoin(all_video_dir, f"{all_keyframes[keypose_idx]}_expert", "low_dim_obs_end.pkl")
                        if not os.path.exists(curr_obs_path) or not os.path.exists(next_obs_path):
                            print(f"Observation files not found: {curr_obs_path}, {next_obs_path}")
                            continue
                        curr_obs = pickle.load(open(curr_obs_path, "rb"))
                        next_obs = pickle.load(open(next_obs_path, "rb"))
                        act = get_action(curr=curr_obs, next=next_obs)
                        
                        img_obs_paths = [pjoin(file_dir, view, "end.png")]
                        if obs_window_size >= 2:
                            img_obs_paths.append(pjoin(file_dir, view, "begin.png"))
                        if obs_window_size >= 3:
                            for i in range(1, obs_window_size - 1):
                                if keypose_idx - i < 0:
                                    break
                                img_obs_paths.append(
                                    pjoin(all_video_dir, f"{all_keyframes[keypose_idx - i]}_expert", view, "begin.png")
                                )
                        img_obs_paths.reverse()
                        if any([not os.path.exists(p) for p in img_obs_paths]):
                            print(f"Image observation files not found: {img_obs_paths}")
                            continue
                        # for each image, save it to the output directory by creating a link
                        new_img_paths = []
                        for img_obs_path in img_obs_paths:
                            img_obs_path = img_obs_path.replace(data_dir, "")
                            img_obs_path = img_obs_path[1:] if img_obs_path.startswith("/") else img_obs_path
                            img_id = "_".join(img_obs_path.split("/"))
                            new_img_path = pjoin(img_save_dir, img_id)
                            if not os.path.exists(new_img_path):
                                os.symlink(pjoin(data_dir, img_obs_path), new_img_path)
                            new_img_paths.append(img_id)
                            
                        in_prompt = ""
                        for _ in range(len(new_img_paths)):
                            in_prompt += "<image>"
                        lang_goal = info["lang_goal"]
                        in_prompt += f"What action should the robot take to {lang_goal}?"
                        
                        if "failure_reason_gpt" not in info or "correction_instruction_gpt" not in info:
                            print(f"Failure reason or correction instruction not found in info: {filename}")
                            continue
                        answer = "<|reason_start|>"
                        answer += f"Previous action is unsuccessful. "
                        answer += f"The action fails because {info['failure_reason_gpt'].lower()} "
                        answer += f"To correct the action, the robot should {info['correction_instruction_gpt'].lower()}"
                        answer += "<|reason_end|>"
                        # # tempt just write action as a string of numbers
                        # act = act.tolist()
                        # action_encoded = " ".join([str(a) for a in act])
                        # answer += action_encoded
                        answer += action2str(act)
                        
                        metadata.append({
                            "id": video_cnt,
                            "image": new_img_paths[0] if len(new_img_paths) <= 1 else new_img_paths,
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": in_prompt
                                },
                                {
                                    "from": "gpt",
                                    "value": answer
                                }
                            ]
                        })
                    else:
                        assert False, f"Unknown filename: {filename}"
                            
                        
                    video_cnt += 1
                    
                    print(f"video {video_cnt} | split {split} | task {task} | episode {episode} | filename {filename}")
                        

    metadata_path = pjoin(output_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in metadata:
            json.dump(item, f, ensure_ascii=False, indent=4)
            f.write("\n")
    print(f"Metadata saved to {metadata_path}")
    print(f"Total videos processed: {video_cnt}")     
