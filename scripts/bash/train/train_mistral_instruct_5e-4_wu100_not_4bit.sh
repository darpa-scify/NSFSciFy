REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python "${REPO_ROOT}/scripts/train.py" --model_name unsloth/mistral-7b-instruct-v0.3 --lr 5e-4 \
--max_seq_length 2048 --max_steps -1 --num_epochs 3 \
--prompt_mode tech2nontech_instruct_user_assistant --warmup_steps 100 \
--disable_4bit