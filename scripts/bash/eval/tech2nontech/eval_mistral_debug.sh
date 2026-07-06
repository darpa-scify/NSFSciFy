#!/bin/bash
# List of models to evaluate
models=(
  "unsloth/mistral-7b-instruct-v0.3"
  "models/lora_model_mistral-7b-instruct-v0.3_r128_la64_lr1e-05_e3_s-1_wu100_bs2_schlinear_optadamw_8bit_emb_lm_emb_lm"
  # "models/lora_model_mistral-7b-instruct-v0.3_r16_la16_lr1e-05_e5_s-1_wu100_bs2_schlinear_optadamw_8bit"
  # "models/lora_model_mistral-7b-instruct-v0.3_r128_la64_lr1e-05_e5_s-1_wu100_bs2_schlinear_optadamw_8bit_emb_lm"
  # "models/lora_model_mistral-7b-instruct-v0.3_r128_la64_lr1e-05_e3_s-1_wu100_bs2_schlinear_optadamw_8bit_emb_lm"
  # "models/lora_model_mistral-7b-instruct-v0.3_lr5e-05_e3_s-1_wu100_bs2_schlinear_optadamw_8bit"
)

# Root directory for dataset and model files
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi
ROOT_DIR="${NSFSCIFY_ROOT_DIR:-${REPO_ROOT}}"
# Prompt mode (adjust as needed)
PROMPT_MODE="tech2nontech_instruct_user_assistant"
# Optional: add debug flag
DEBUG_FLAG="--debug"

# Loop over each model and run evaluation
for model in "${models[@]}"; do
    echo "----------------------------------------"
    echo "Evaluating model: ${model}"
    echo "----------------------------------------"
    python "${REPO_ROOT}/scripts/eval.py" \
        --model "${model}" \
        --root_dir "${ROOT_DIR}" \
        --prompt_mode "${PROMPT_MODE}" \
        ${DEBUG_FLAG}
done
