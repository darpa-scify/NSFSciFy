REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python "${REPO_ROOT}/scripts/train.py" --model_name unsloth/Meta-Llama-3.1-8B-Instruct --lr 1e-5 \
--max_seq_length 2048 --max_steps -1 --num_epochs 3 --prompt_mode technontech2claims_instruct \
--r 128 \
--warmup_steps 100 \
--disable_4bit \
--output_suffix _emb_lm \
--lora_alpha 64 \
--target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj embed_tokens lm_head