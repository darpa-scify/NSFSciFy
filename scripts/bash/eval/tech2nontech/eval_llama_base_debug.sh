#!/bin/bash
# List of models to evaluate
models=(
  "unsloth/Meta-Llama-3.1-8B"
  # "models/lora_model_Meta-Llama-3.1-8B_r128_la64_lr1e-05_e3_s-1_wu100_bs2_schlinear_optadamw_8bit_emb_lm"
  "models/lora_model_Meta-Llama-3.1-8B_r128_la64_lr1e-05_e3_s-1_wu100_bs2_schlinear_optadamw_8bit_emb_lm_emb_lm"
)

# Root directory for dataset and model files
ROOT_DIR="/shared_data0/weiqiuy/nsf-awards"
# Prompt mode (adjust as needed)
PROMPT_MODE="tech2nontech"
# Optional: add debug flag
DEBUG_FLAG="--debug"

# Loop over each model and run evaluation
for model in "${models[@]}"; do
    echo "----------------------------------------"
    echo "Evaluating model: ${model}"
    echo "----------------------------------------"
    python scripts/eval.py \
        --model "${model}" \
        --root_dir "${ROOT_DIR}" \
        --prompt_mode "${PROMPT_MODE}" \
        ${DEBUG_FLAG}
done
