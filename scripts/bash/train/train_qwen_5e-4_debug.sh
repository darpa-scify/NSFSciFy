REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
python "${REPO_ROOT}/scripts/train.py" --model_name unsloth/Qwen2.5-7B --output_suffix lr5e-4 --lr 5e-4 \
--max_seq_length 1024 --max_steps 60 --num_epochs -1