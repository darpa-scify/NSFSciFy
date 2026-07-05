#!/bin/bash
# List of models to evaluate
models=(
  "models/lora_model_mistral-7b-instruct-v0.3.text2ip_instruct_user_assistant.r128_la64_lr1e-05_e3_s-1_wu100_bs2_schlinear_optadamw_8bit_emb_lm"
  # "models/lora_model_mistral-7b-instruct-v0.3.text2ip_instruct_user_assistant.matsci_and_20k.r128_la64_lr1e-05_e3_s-1_wu100_bs2_schlinear_optadamw_8bit_emb_lm"
  # "unsloth/mistral-7b-instruct-v0.3"
  # "models/lora_model_mistral-7b-instruct-v0.3.technontech2claims_instruct_user_assistant.r128_la64_lr1e-05_e3_s-1_wu100_bs2_schlinear_optadamw_8bit_emb_lm"
)

# Root directory for dataset and model files
ROOT_DIR="/shared_data0/weiqiuy/nsf-awards"
# Prompt mode (adjust as needed)
PROMPT_MODE="text2ip_instruct_user_assistant"
# Task name
TASK="technontech2ip"
# Optional: add debug flag
# DEBUG_FLAG="--debug"

# Loop over each model and run evaluation
for model in "${models[@]}"; do
    echo "----------------------------------------"
    echo "Evaluating model: ${model}"
    echo "----------------------------------------"
    python scripts/gen.py \
        --model "${model}" \
        --root_dir "${ROOT_DIR}" \
        --save_dir "results_gen_bsz1" \
        --prompt_mode "${PROMPT_MODE}" \
        --task "${TASK}" \
        --dataset "matsci_and_20k" \
        --batch_size 1
        # \
        # --gen_only 
        # \
        # ${DEBUG_FLAG}
done
