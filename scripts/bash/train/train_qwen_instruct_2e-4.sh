REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python "${REPO_ROOT}/scripts/train.py" --model_name unsloth/Qwen2.5-7B-Instruct --output_suffix .lr2e-4 --lr 2e-4 \
--max_seq_length 2048 --max_steps -1 --num_epochs 3 --prompt_mode tech2nontech_instruct