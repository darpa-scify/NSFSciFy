# NSFSciFy

Accompanying code for NSF-SciFy: Mining the NSF Awards Database for Scientific Claims.

This repo contains the reproducible code for filtering NSF award data into the final datasets, training LoRA models, generating predictions, and evaluating those predictions with automatic and LLM-judge metrics.

## What To Use

Data processing:

- [`notebooks/filter_dup_data_all.ipynb`](https://github.com/darpa-scify/NSFSciFy/blob/main/notebooks/filter_dup_data_all.ipynb): filters the 20K NSF awards data into the final train/validation/test JSONL splits.
- [`notebooks/filter_dup_data_matsci.ipynb`](https://github.com/darpa-scify/NSFSciFy/blob/main/notebooks/filter_dup_data_matsci.ipynb): filters the materials-science/DMR awards data into the final train/validation/test JSONL splits.

Training and generation:

- [`notebooks/matsci_20k_single_prediction_demo.ipynb`](https://github.com/darpa-scify/NSFSciFy/blob/main/notebooks/matsci_20k_single_prediction_demo.ipynb): minimal Hugging Face demo that loads a released matsci+20K adapter, predicts on one dataset entry, and evaluates that prediction.
- [`src/data_utils.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/src/data_utils.py): loads local JSONL data and formats task prompts.
- [`scripts/train.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/scripts/train.py): main Unsloth/TRL SFT LoRA training entry point.
- [`scripts/gen.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/scripts/gen.py): generation-only entry point.
- [`scripts/bash/train/`](https://github.com/darpa-scify/NSFSciFy/tree/main/scripts/bash/train): shell launchers for the training runs.

Evaluation:

- [`scripts/eval.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/scripts/eval.py): generation plus metric computation.
- [`src/eval_utils.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/src/eval_utils.py): generation helpers plus BERTScore, ROUGE, and BLEU evaluation.
- [`src/claims_utils.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/src/claims_utils.py): LLM-judge precision, recall, and F-score for verifiable claims and investigation proposals.
- [`src/text_utils.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/src/text_utils.py), [`src/model_utils.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/src/model_utils.py), [`src/openai_request_utils.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/src/openai_request_utils.py), and [`src/vis_utils.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/src/vis_utils.py): support utilities.
- [`scripts/bash/eval/`](https://github.com/darpa-scify/NSFSciFy/tree/main/scripts/bash/eval): shell launchers for evaluation runs.
- [`notebooks/get_scores.ipynb`](https://github.com/darpa-scify/NSFSciFy/blob/main/notebooks/get_scores.ipynb), [`notebooks/show_results.ipynb`](https://github.com/darpa-scify/NSFSciFy/blob/main/notebooks/show_results.ipynb), [`notebooks/show_results_llama.ipynb`](https://github.com/darpa-scify/NSFSciFy/blob/main/notebooks/show_results_llama.ipynb), [`notebooks/show_results_tech-nontech2claims.ipynb`](https://github.com/darpa-scify/NSFSciFy/blob/main/notebooks/show_results_tech-nontech2claims.ipynb), [`notebooks/show_results_tech-nontech2ip.ipynb`](https://github.com/darpa-scify/NSFSciFy/blob/main/notebooks/show_results_tech-nontech2ip.ipynb), and [`notebooks/metrics/`](https://github.com/darpa-scify/NSFSciFy/tree/main/notebooks/metrics): result aggregation and display notebooks.

Generated data, model checkpoints, result files, W&B logs, caches, and the original Hugging Face upload notebook are not included.

## Data

The filtering notebooks document how the released datasets were made. Most users should load the released data from Hugging Face rather than rerunning filtering.

Released datasets:

- `darpa-scify/nsf-scify-matsci`: materials-science/DMR NSF awards.
- `darpa-scify/nsf-scify-20k`: 20K all-awards NSF-SciFy subset.

Both datasets provide `train`, `validation`, and `test` splits. They include the generated `non_technical_abstract`, `verifiable_claims`, and `investigation_proposals` fields used by the training and evaluation scripts.

Runnable Hugging Face examples are also available in [`notebooks/hf_usage_examples.ipynb`](https://github.com/darpa-scify/NSFSciFy/blob/main/notebooks/hf_usage_examples.ipynb). For the shortest end-to-end model demo, use [`notebooks/matsci_20k_single_prediction_demo.ipynb`](https://github.com/darpa-scify/NSFSciFy/blob/main/notebooks/matsci_20k_single_prediction_demo.ipynb).

Load the materials-science data:

```python
from datasets import load_dataset

ds = load_dataset("darpa-scify/nsf-scify-matsci")
train_ds = ds["train"]
val_ds = ds["validation"]
test_ds = ds["test"]
```

Load the 20K data:

```python
from datasets import load_dataset

ds = load_dataset("darpa-scify/nsf-scify-20k")
train_ds = ds["train"]
val_ds = ds["validation"]
test_ds = ds["test"]
```

Load the combined `matsci_and_20k` training data by loading the two datasets separately and concatenating matching splits:

```python
from datasets import concatenate_datasets, load_dataset

matsci = load_dataset("darpa-scify/nsf-scify-matsci")
twenty_k = load_dataset("darpa-scify/nsf-scify-20k")

train_ds = concatenate_datasets([matsci["train"], twenty_k["train"]])
val_ds = concatenate_datasets([matsci["validation"], twenty_k["validation"]])
test_ds = concatenate_datasets([matsci["test"], twenty_k["test"]])
```

The copied training/evaluation scripts currently call [`src/data_utils.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/src/data_utils.py), which expects local JSONL files under `data/`:

- materials science: `data/dmr_nsf_awards_with_claims_filtered_train_v01.jsonl`, `data/dmr_nsf_awards_with_claims_filtered_val_v01.jsonl`, and `data/dmr_nsf_awards_with_claims_filtered_test_v01.jsonl`;
- 20K: `data/20K_nsf_awards_with_claims_filtered_train.jsonl`, `data/20K_nsf_awards_with_claims_filtered_val.jsonl`, and `data/20K_nsf_awards_with_claims_filtered_test.jsonl`.

To use Hugging Face directly with the existing scripts, replace the local loader in [`src/data_utils.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/src/data_utils.py) with a `load_dataset(...)` call that returns `train`, `validation`, and `test`:

```python
from datasets import load_dataset

def get_nsf_data_raw(ROOT_DIR):
    ds = load_dataset("darpa-scify/nsf-scify-matsci")
    return ds["train"], ds["validation"], ds["test"]
```

Use `darpa-scify/nsf-scify-20k` in `get_nsf_data_20k_raw(...)`. For `get_nsf_data_matsci_and_20k_raw(...)`, load both repositories and concatenate each split as shown above.

## Released Models

The released models are LoRA adapters on top of `unsloth/mistral-7b-instruct-v0.3`. Use the adapter repository name anywhere the scripts accept `--model`.

Materials-science/DMR adapters:

- `darpa-scify/nsf-scify-matsci-nta`: technical abstract to non-technical abstract.
- `darpa-scify/nsf-scify-matsci-claims`: abstract text to verifiable claims.
- `darpa-scify/nsf-scify-matsci-ip`: abstract text to investigation proposals.

Materials-science plus 20K adapters:

- `darpa-scify/nsf-scify-matsci-20k-nta`: technical abstract to non-technical abstract.
- `darpa-scify/nsf-scify-matsci-20k-claims`: abstract text to verifiable claims.
- `darpa-scify/nsf-scify-matsci-20k-ip`: abstract text to investigation proposals.

Example inference loading:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="darpa-scify/nsf-scify-matsci-20k-claims",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

The same IDs can be passed directly to [`scripts/gen.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/scripts/gen.py) or [`scripts/eval.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/scripts/eval.py).

## Training

Run [`scripts/train.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/scripts/train.py) from the repo root:

```bash
python scripts/train.py \
  --ROOT_DIR . \
  --dataset_name matsci \
  --model_name unsloth/mistral-7b-instruct-v0.3-bnb-4bit \
  --prompt_mode tech2nontech_instruct_user_assistant \
  --lr 1e-5 \
  --num_epochs 3 \
  --warmup_steps 100 \
  --batch_size 2 \
  --r 128 \
  --lora_alpha 64
```

Use `--dataset_name matsci`, `20k`, or `matsci_and_20k`. Common prompt modes are `tech2nontech_instruct_user_assistant`, `text2claims_instruct_user_assistant`, and `text2ip_instruct_user_assistant`. Exact experiment launchers are under [`scripts/bash/train/`](https://github.com/darpa-scify/NSFSciFy/tree/main/scripts/bash/train).

LoRA adapters are saved under `models/lora_model_<run-name>/`.

## Generation

Use [`scripts/gen.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/scripts/gen.py) when you only want predictions:

```bash
python scripts/gen.py \
  --root_dir . \
  --model darpa-scify/nsf-scify-matsci-20k-nta \
  --dataset matsci_and_20k \
  --task tech2nontech \
  --prompt_mode tech2nontech_instruct_user_assistant \
  --batch_size 1
```

Predictions are written under `results_gen/` by default.

## Evaluation

Use [`scripts/eval.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/scripts/eval.py) for prediction plus metrics:

```bash
python scripts/eval.py \
  --root_dir . \
  --model darpa-scify/nsf-scify-matsci-20k-nta \
  --dataset matsci_and_20k \
  --task tech2nontech \
  --prompt_mode tech2nontech_instruct_user_assistant \
  --batch_size 1
```

Tasks:

- `tech2nontech`: generate non-technical abstracts from technical abstracts.
- `technontech2claims`: generate verifiable claims.
- `technontech2ip`: generate investigation proposals.
- `technontech2claimsip`: combined claims and investigation proposals; generation-only support exists, but full scoring is not implemented in [`scripts/eval.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/scripts/eval.py).

Metrics:

- `tech2nontech` uses BERTScore, ROUGE, and BLEU in [`src/eval_utils.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/src/eval_utils.py).
- Claims and investigation-proposal tasks use LLM-judge pairwise comparison in [`src/claims_utils.py`](https://github.com/darpa-scify/NSFSciFy/blob/main/src/claims_utils.py) to compute precision, recall, and F-score.

For LLM-judge metrics, configure `OPENAI_API_KEY` in the environment or in a repo-root `.env` file.

Use matching models, datasets, tasks, and prompt modes:

- NTA models: `--task tech2nontech`, `--prompt_mode tech2nontech_instruct_user_assistant`.
- Claims models: `--task technontech2claims`, `--prompt_mode text2claims_instruct_user_assistant`.
- Investigation-proposal models: `--task technontech2ip`, `--prompt_mode text2ip_instruct_user_assistant`.
- Materials-science models usually pair with `--dataset matsci`; materials-science plus 20K models usually pair with `--dataset matsci_and_20k` or can be evaluated on either constituent dataset.

## Portability Notes

Python entry points resolve data, model checkpoints, caches, and output directories relative to the repository root by default. To override local settings, copy `.env.example` to `.env`; the scripts load it when present. Relative paths in `.env`, such as `HF_HOME=.cache/huggingface`, are resolved from the repository root.

Common `.env` values:

```bash
NSFSCIFY_ROOT_DIR=.
HF_HOME=.cache/huggingface
WANDB_PROJECT=nsf
OPENAI_API_KEY=
```

The original `notebooks/upload2hf.ipynb` was not copied because it contains a hard-coded Hugging Face token.
