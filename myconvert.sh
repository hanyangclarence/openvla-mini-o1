export CUDA_VISIBLE_DEVICES=0

python vla-scripts/extern/convert_openvla_weights_to_hf.py \
    --openvla_model_path_or_id "logs/prism-qwen25-dinosiglip-224px+0_5b+mx-roboverse+n1+b32+x7" \
    --output_hf_model_local_path "runs/prism-qwen25-dinosiglip-224px+0_5b+mx-roboverse-57500" \
    --output_hf_model_hub_path "szang18/openvla-roboverse"