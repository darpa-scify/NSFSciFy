echo "Running all training scripts for technontechclaims2ip"
echo "Running qwen_instruct_1e-5_wu100_r128_new"
bash scripts/bash/train/technontech2claimsip/train_qwen_instruct_1e-5_wu100_r128_new.sh
echo "Running mistral_instruct_1e-5_wu100_r128_new"
bash scripts/bash/train/technontech2claimsip/train_mistral_instruct_1e-5_wu100_r128_new.sh