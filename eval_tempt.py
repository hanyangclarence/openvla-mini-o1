from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import pdb
from peft import LoraConfig, PeftModel
from typing import Optional
import glob
import os

pdb.set_trace()

saved_model_directory = "openvla/openvla-7b"
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

all_transitions = glob.glob("/gpfs/yanghan/data/runs_vla_data/val/*/0/video/*")
for path in all_transitions:
    if "expert" in path:
        obs_path = f"{path}/front_rgb/begin.png"
    elif "perturb" in path:
        obs_path = f"{path}/front_rgb/end.png"
    else:
        continue
    json_path = f"{path}/info.json"
    if not os.path.exists(obs_path) or not os.path.exists(json_path):
        continue
    
    with open(json_path, "r") as f:
        json_data = f.read()
        task_instruction = json_data['lang_goal']
    
    image = Image.open(obs_path).convert("RGB")
    prompt = f"In: What action should the robot take to {task_instruction}?\nOut:"
    
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    generated_ids = model.generate(**inputs, max_new_tokens=200)
    
    output_str = processor.tokenizer.decode(
        generated_ids.squeeze(0).cpu().numpy().tolist(),
        skip_special_tokens=False
    )
    
    print(f"Path: {path}")
    print(output_str)

# obs_path = "/gpfs/yanghan/data/runs_vla_data/val/meat_off_grill/0/video/59_perturb_0/front_rgb/end.png"
# task_instruction = "take the steak off the grill"

# image = Image.open(obs_path).convert("RGB")
# prompt = f"In: What action should the robot take to {task_instruction}?\nOut:"

# inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
# generated_ids = model.generate(**inputs, max_new_tokens=100)

# output_str = processor.tokenizer.decode(
#     generated_ids.squeeze(0).cpu().numpy().tolist(),
#     skip_special_tokens=False
# )

# print(output_str)