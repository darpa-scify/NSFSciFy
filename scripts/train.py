from unsloth import FastLanguageModel
import torch
import sys
sys.path.append('src')
from data_utils import get_nsf_data_matsci_and_20k_proc, get_nsf_data_20k_proc, get_nsf_data_proc
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import os
import argparse

os.environ["WANDB_PROJECT"] = 'nsf'
os.environ["HF_HOME"] = '/shared_data0/hf_cache'

def main():

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--ROOT_DIR', type=str, default='.')
        parser.add_argument('--max_seq_length', type=int, default=2048)
        parser.add_argument('--dataset_name', type=str, default='matsci', choices=['matsci_and_20k', '20k', 'matsci'])
        parser.add_argument('--model_name', type=str, default='unsloth/Qwen2.5-7B')
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--num_epochs', type=int, default=1)
        parser.add_argument('--max_steps', type=int, default=-1)
        parser.add_argument('--output_suffix', type=str, default='')
        parser.add_argument('--prompt_mode', type=str, choices=['tech2nontech', 'tech2nontech_instruct', 'tech2nontech_instruct_user_assistant',
            'technontech2claims', 'technontech2claims_instruct', 'technontech2claims_instruct_user_assistant', 'text2claims_instruct', 'text2claims_instruct_user_assistant',
            'technontech2ip', 'technontech2ip_instruct', 'technontech2ip_instruct_user_assistant', 'text2ip_instruct', 'text2ip_instruct_user_assistant',
            'technontech2claimsip', 'technontech2claimsip_instruct', 'technontech2claimsip_instruct_user_assistant'], default='tech2nontech')
        parser.add_argument('--warmup_steps', type=int, default=5)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--scheduler_type', type=str, default='linear')
        parser.add_argument('--optimizer_type', type=str, default='adamw_8bit')
        parser.add_argument('--disable_bfloat16', action='store_true')
        parser.add_argument('--disable_4bit', action='store_true')
        parser.add_argument('--r', type=int, default=16)
        parser.add_argument('--lora_alpha', type=int, default=16)
        parser.add_argument('--target_modules', type=str, nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

        return parser.parse_args()

    args = get_args()

    print("All arguments:", vars(args))

    ROOT_DIR = '.'


    max_seq_length = args.max_seq_length # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = not args.disable_4bit # Use 4bit quantization to reduce memory usage. Can be False.
    model_name = args.model_name # Choose any model from the list above or any other model from Hugging Face!
    lr = args.lr
    num_epochs = args.num_epochs
    max_steps = args.max_steps # Set this for 1 full training run. -1 for full dataset.
    output_suffix = args.output_suffix
    prompt_mode = args.prompt_mode
    warmup_steps = args.warmup_steps
    batch_size = args.batch_size
    scheduler_type = args.scheduler_type
    optimizer_type = args.optimizer_type
    disable_bfloat16 = args.disable_bfloat16
    dataset_name = args.dataset_name
    r = args.r
    lora_alpha = args.lora_alpha

    save_name = f"{model_name.split('/')[-1]}.{prompt_mode}.{dataset_name}.r{r}_la{lora_alpha}_lr{lr}_e{num_epochs}_s{max_steps}_wu{warmup_steps}_bs{batch_size}_sch{scheduler_type}_opt{optimizer_type}{output_suffix}"

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
        "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
        "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
        "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
    ] # More models at https://huggingface.co/unsloth


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    if dataset_name == 'matsci_and_20k':
        train_dataset, val_dataset, test_dataset = get_nsf_data_matsci_and_20k_proc(ROOT_DIR, tokenizer, mode=prompt_mode)
    elif dataset_name == '20k':
        train_dataset, val_dataset, test_dataset = get_nsf_data_20k_proc(ROOT_DIR, tokenizer, mode=prompt_mode)
    elif dataset_name == 'matsci':
        train_dataset, val_dataset, test_dataset = get_nsf_data_proc(ROOT_DIR, tokenizer, mode=prompt_mode)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = 4,
            warmup_steps = warmup_steps,
            num_train_epochs = num_epochs, # Set this for 1 full training run.
            eval_strategy = 'steps',
            eval_steps = 250,
            max_steps = max_steps,
            learning_rate = lr,
            fp16 = disable_bfloat16 or not is_bfloat16_supported(),
            bf16 = not disable_bfloat16 and is_bfloat16_supported(),
            logging_steps = 1,
            optim = optimizer_type,
            weight_decay = 0.01,
            lr_scheduler_type = scheduler_type,
            seed = 3407,
            output_dir = os.path.join('models', save_name),
            report_to = "wandb", # Use this for WandB etc
            run_name=save_name,
        ),
    )

    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    #@title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.save_pretrained(os.path.join(ROOT_DIR, f"models/lora_model_{save_name}")) # Local saving
    tokenizer.save_pretrained(os.path.join(ROOT_DIR, f"models/lora_model_{save_name}"))


if __name__ == '__main__':
    main()