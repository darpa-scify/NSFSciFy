
from datasets import Dataset, DatasetDict, concatenate_datasets
import inspect
import json
import os
from datasets import load_dataset


from typing import List, Union
from datasets import concatenate_datasets, Dataset, Value
from typing import List

def concat_datasets_drop_mismatches(
    datasets: List[Dataset]
) -> Dataset:
    """
    Concatenate a list of HuggingFace Datasets, first dropping any columns
    that are either:
      - Present in only a subset of the datasets, or
      - Present in all datasets but with mismatched Feature types.

    Args:
        datasets: List of `datasets.Dataset` objects to concatenate.

    Returns:
        A single `Dataset` with only the aligned columns, obtained by
        dropping the problematic ones from each input dataset before concat.
    """
    # 1) Gather feature dicts from each dataset
    feature_dicts = [ds.features for ds in datasets]

    # 2) Compute which columns are common to all vs. only in some
    cols_sets = [set(feat) for feat in feature_dicts]
    common_cols = set.intersection(*cols_sets)
    all_cols = set.union(*cols_sets)
    only_in_some = all_cols - common_cols

    # 3) Among the common columns, find those whose types disagree
    mismatched = []
    for col in common_cols:
        types = [feat_dict[col] for feat_dict in feature_dicts]
        if any(t != types[0] for t in types[1:]):
            mismatched.append(col)

    # 4) Final list of columns to drop
    to_drop = list(only_in_some | set(mismatched))
    if to_drop:
        print(f"[concat_datasets_drop_mismatches] dropping columns: {to_drop}")

    # 5) Remove only existing columns from each dataset
    cleaned = []
    for ds in datasets:
        cols_in_ds = set(ds.column_names)
        cols_to_rm = [c for c in to_drop if c in cols_in_ds]
        if cols_to_rm:
            ds = ds.remove_columns(cols_to_rm)
        cleaned.append(ds)

    # 6) Finally concatenate
    return concatenate_datasets(cleaned)



def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def load_jsonl_to_datasetdict(file_path, split_name="train"):
    """
    Load a JSONL file and convert it into a Hugging Face DatasetDict.

    Args:
        file_path (str): Path to the JSONL file.
        split_name (str): Name of the split (default is "train").

    Returns:
        DatasetDict: A Hugging Face DatasetDict containing the data.
    """
    # Load JSONL data

    if file_path.endswith('.json'):
        with open(file_path, 'rt') as file:
            data = json.load(file)
    else:
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))

    # Create a Hugging Face Dataset from the list of dictionaries
    dataset = Dataset.from_list(data)

    # Wrap it into a DatasetDict with the specified split name
    dataset_dict = DatasetDict({split_name: dataset})
    return dataset_dict


def get_nsf_data_raw(ROOT_DIR):
    """
    Loads pre-split NSF award data from JSONL files for train, validation, and test splits.
    
    Parameters:
        ROOT_DIR (str): The root directory containing the data folder.
        save (bool): If True, the loaded datasets will be saved to disk in JSON format.
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Define file paths for the pre-split datasets
    train_file = os.path.join(ROOT_DIR, 'data', 'dmr_nsf_awards_with_claims_filtered_train_v01.jsonl')
    val_file   = os.path.join(ROOT_DIR, 'data', 'dmr_nsf_awards_with_claims_filtered_val_v01.jsonl')
    test_file  = os.path.join(ROOT_DIR, 'data', 'dmr_nsf_awards_with_claims_filtered_test_v01.jsonl')
    
    # Load each dataset using your helper function.
    # It is assumed that load_jsonl_to_datasetdict returns a dictionary
    # with a single key corresponding to the given split.
    train_dataset = load_jsonl_to_datasetdict(train_file, split_name="train")["train"]
    val_dataset   = load_jsonl_to_datasetdict(val_file, split_name="validation")["validation"]
    test_dataset  = load_jsonl_to_datasetdict(test_file, split_name="test")["test"]
    
    # Optionally, inspect the first entry of each dataset
    print("Train dataset keys:", train_dataset[0].keys())
    print("Validation dataset keys:", val_dataset[0].keys())
    print("Test dataset keys:", test_dataset[0].keys())
    
    return train_dataset, val_dataset, test_dataset

def get_nsf_data_20k_raw(ROOT_DIR):
    """
    Loads pre-split NSF award data from JSONL files for train, validation, and test splits.
    
    Parameters:
        ROOT_DIR (str): The root directory containing the data folder.
        save (bool): If True, the loaded datasets will be saved to disk in JSON format.
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Define file paths for the pre-split datasets
    train_file = os.path.join(ROOT_DIR, 'data', '20K_nsf_awards_with_claims_filtered_train.jsonl')
    val_file   = os.path.join(ROOT_DIR, 'data', '20K_nsf_awards_with_claims_filtered_val.jsonl')
    test_file  = os.path.join(ROOT_DIR, 'data', '20K_nsf_awards_with_claims_filtered_test.jsonl')
    
    # Load each dataset using your helper function.
    # It is assumed that load_jsonl_to_datasetdict returns a dictionary
    # with a single key corresponding to the given split.
    train_dataset = load_jsonl_to_datasetdict(train_file, split_name="train")["train"]
    val_dataset   = load_jsonl_to_datasetdict(val_file, split_name="validation")["validation"]
    test_dataset  = load_jsonl_to_datasetdict(test_file, split_name="test")["test"]
    
    # Optionally, inspect the first entry of each dataset
    print("Train dataset keys:", train_dataset[0].keys())
    print("Validation dataset keys:", val_dataset[0].keys())
    print("Test dataset keys:", test_dataset[0].keys())
    
    return train_dataset, val_dataset, test_dataset

def get_nsf_data_all_raw(ROOT_DIR):
    """
    Loads pre-split NSF award data from JSONL files for train, validation, and test splits.
    
    Parameters:
        ROOT_DIR (str): The root directory containing the data folder.
        save (bool): If True, the loaded datasets will be saved to disk in JSON format.
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Define file paths for the pre-split datasets
    all_file = os.path.join(ROOT_DIR, 'data', 'nsf_awards-0325.json')
    
    # Load each dataset using your helper function.
    # It is assumed that load_jsonl_to_datasetdict returns a dictionary
    # with a single key corresponding to the given split.
    all_dataset = load_jsonl_to_datasetdict(all_file, split_name="train")["train"]
    
    # Optionally, inspect the first entry of each dataset
    print("All dataset keys:", all_dataset[0].keys())
    
    return all_dataset


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

instructions_dict = {
    'tech2nontech': """Given a technical abstract as an input, please output a non-technical abstract as a response.
    
    Input format:
    TECHNICAL ABSTRACT:
    <technical abstract>

    Output format:
    NON-TECHNICAL ABSTRACT:
    <non-technical abstract>
    """,
    'tech2nontech_instruct': """You're an expert technical writer at rewriting technical to non-technical abstract. Your job is to output a non-technical abstract that is a rewrite of the technical abstract the user gives you.
    
    Input format:
    TECHNICAL ABSTRACT:
    <technical abstract>

    Output format:
    NON-TECHNICAL ABSTRACT:
    <non-technical abstract>
    """,
    'technontech2claims': """Given a technical abstract and a non-technical abstract as inputs, please output a list of verifiable claims as a response.
    
    Input format:
    TECHNICAL ABSTRACT:
    <technical abstract>

    NON-TECHNICAL ABSTRACT:
    <non-technical abstract>

    Output format:
    CLAIMS:
    ```json
    [
        "<claim 1>",
        "<claim 2>",
    ]
    ```
    """,
    'technontech2claims_instruct': """You're an expert technical writer at extracting claims from technical and non-technical abstracts. Your job is to output a list of verifiable claims given a technical abstract and a non-technical abstract.
    
    Input format:
    TECHNICAL ABSTRACT:
    <technical abstract>

    NON-TECHNICAL ABSTRACT:
    <non-technical abstract>

    Output format:
    CLAIMS:
    ```json
    [
        "<claim 1>",
        "<claim 2>",
    ]
    ```
    """,
    'text2claims_instruct': """You're an expert at extracting claims from text. Your job is to output a list of claims given a piece of text.

TEXT:
<text>

CLAIMS:
```json
[
    "<claim 1>",
    "<claim 2>",
    ...
]
```""",
    'technontech2ip': """Given a technical abstract and a non-technical abstract as inputs, please output a list of investigation proposals as a response.
    
    Input format:
    TECHNICAL ABSTRACT:
    <technical abstract>

    NON-TECHNICAL ABSTRACT:
    <non-technical abstract>

    Output format:
    INVESTIGATION PROPOSALS:
    ```json
    [
        "<claim 1>",
        "<claim 2>",
    ]
    ```
    """,
    'technontech2ip_instruct': """You're an expert technical writer at making investigation proposals from technical and non-technical abstracts. Your job is to output a list of investigation proposals given a technical abstract and a non-technical abstract.
    
    Input format:
    TECHNICAL ABSTRACT:
    <technical abstract>

    NON-TECHNICAL ABSTRACT:
    <non-technical abstract>

    Output format:
    INVESTIGATION PROPOSALS:
    ```json
    [
        "<claim 1>",
        "<claim 2>",
    ]
    ```
    """,
    'text2ip_instruct': """You're an expert at extracting investigation proposals from text. Your job is to output a list of investigation proposals given a piece of text.

TEXT:
<text>

INVESTIGATION PROPOSALS:
```json
[
    "<investigation proposal 1>",
    "<investigation proposal 2>",
    ...
]
```""",
    'technontech2claimsip': """Given a technical abstract and a non-technical abstract as inputs, please output a dictionary which contains a list of claims and a list of investigation proposals as a response.
    
    Input format:
    TECHNICAL ABSTRACT:
    <technical abstract>

    NON-TECHNICAL ABSTRACT:
    <non-technical abstract>

    Output format:
    ```json
    {{
        "claims": [
            "<claim 1>",
            "<claim 2>",
        ],
        "investigation_proposals": [
            "<investigation proposal 1>",
            "<investigation proposal 2>",
        ]
    }}
    ```
    """,
    'technontech2claimsip_instruct': """You're an expert technical writer at extracting claims and making investigation proposals from technical and non-technical abstracts. Your job is to output a dictionary which contains a list of claims and a list of investigation proposals given a technical abstract and a non-technical abstract.
    
    Input format:
    TECHNICAL ABSTRACT:
    <technical abstract>

    NON-TECHNICAL ABSTRACT:
    <non-technical abstract>

    Output format:
    ```json
    {{
        "claims": [
            "<claim 1>",
            "<claim 2>",
        ],
        "investigation_proposals": [
            "<investigation proposal 1>",
            "<investigation proposal 2>",
        ]
    }}
    ```
    """
}

def formatting_prompts_func_tech2nontech(examples, tokenizer, eval_mode=False):
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

    # tech2nontech
    instructions = instructions_dict['tech2nontech'] #examples["instruction"]
    inputs       = example["technical_abstract"]
    outputs      = example["non_technical_abstract"]
    texts = []

    if isinstance(examples, list):
        for instruction, input, output in zip(instructions, inputs, outputs):
            if not eval_mode:
                text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            else:
                text = alpaca_prompt.format(instruction, input, '') + EOS_TOKEN
            # text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
    else:
        if not eval_mode:
            texts = alpaca_prompt.format(instructions, inputs, outputs) + EOS_TOKEN
        else:
            texts = alpaca_prompt.format(instructions, inputs, '') + EOS_TOKEN
        # texts = alpaca_prompt.format(instructions, inputs, outputs) + EOS_TOKEN

    return { "text" : texts, }


def formatting_prompts_func_tech2nontech_instruct(examples, tokenizer, eval_mode=False):
    system_prompt = instructions_dict['tech2nontech_instruct']
    
    texts = []
    
    def format_chat_prompt(example):
        inputs       = example["technical_abstract"]
        outputs      = example["non_technical_abstract"]
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': inputs},
            # {'role': 'assistant', 'content': outputs}
        ]
        if not eval_mode:
            messages.append({'role': 'assistant', 'content': outputs})
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # We create the string prompt first.
            add_generation_prompt=True  # This adds any necessary special tokens for generation.
        )
        return prompt

    if isinstance(examples, list):
        for example in examples:
            text = format_chat_prompt(example)
            texts.append(text)
    else:
        texts = format_chat_prompt(examples)

    return { "text" : texts, }

def formatting_prompts_func_tech2nontech_instruct_user_assistant(examples, tokenizer, eval_mode=False):
    system_prompt = instructions_dict['tech2nontech_instruct']

    texts = []
    
    def format_chat_prompt(example):
        inputs       = example["technical_abstract"]
        outputs      = example["non_technical_abstract"]
        messages = [
            {'role': 'user', 'content': system_prompt + '\nTECHNICAL ABSTRACT:\n' + inputs + '\nPlease give me the NON-TECHNICAL ABSTRACT alone without other information.'},
            # {'role': 'assistant', 'content': outputs}
        ]

        if not eval_mode:
            messages.append({'role': 'assistant', 'content': outputs})
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # We create the string prompt first.
            add_generation_prompt=True  # This adds any necessary special tokens for generation.
        )
        # Save the formatted prompt in a field (e.g., "text")
        return prompt

    if isinstance(examples, list):
        for example in examples:
            text = format_chat_prompt(example)
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            texts.append(text)
    else:
        texts = format_chat_prompt(examples)

    return { "text" : texts, }


### tech + non-tech -> claims

def formatting_prompts_func_technontech2claims(examples, tokenizer, eval_mode=False):
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

    def get_input_output(example):
        instructions = instructions_dict['technontech2claims']
        inputs_tech_abs       = example["technical_abstract"]
        inputs_non_tech_abs   = example["non_technical_abstract"]
        inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
        outputs               = f'''\nCLAIMS:```json\n{json.dumps(example['verifiable_claims'])}\n```'''
        return instructions, inputs, outputs
    # # tech2nontech
    # instructions          = instructions_dict['technontech2claims'] #examples["instruction"]
    # inputs_tech_abs       = example["technical_abstract"]
    # inputs_non_tech_abs   = example["non_technical_abstract"]
    # inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
    # outputs               = f'''\nCLAIMS:```json\n{json.dumps(example['verifiable_claims'])}\n```'''
    
    texts = []

    if isinstance(examples, list):
        for example in examples:
            instruction, input, output = get_input_output(example)

            if not eval_mode:
                text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            else:
                text = alpaca_prompt.format(instruction, input, '') + EOS_TOKEN

            # text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)

        # for instruction, input, output in zip(instructions, inputs, outputs):
        #     text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        #     texts.append(text)
    else:
        instructions, inputs, outputs = get_input_output(examples)
        if not eval_mode:
            texts = alpaca_prompt.format(instructions, inputs, outputs) + EOS_TOKEN
        else:
            texts = alpaca_prompt.format(instructions, inputs, '') + EOS_TOKEN
        # texts = alpaca_prompt.format(instructions, inputs, outputs) + EOS_TOKEN

    return { "text" : texts, }


def formatting_prompts_func_technontech2claims_instruct(examples, tokenizer, eval_mode=False):
    system_prompt = instructions_dict['technontech2claims_instruct']
    
    texts = []
    
    def format_chat_prompt(example):
        inputs_tech_abs       = example["technical_abstract"]
        inputs_non_tech_abs   = example["non_technical_abstract"]
        inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
        outputs               = f'''CLAIMS:```json\n{json.dumps(example['verifiable_claims'])}\n```'''

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': inputs},
            # {'role': 'assistant', 'content': outputs}
        ]

        if not eval_mode:
            messages.append({'role': 'assistant', 'content': outputs})
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # We create the string prompt first.
            add_generation_prompt=True  # This adds any necessary special tokens for generation.
        )
        return prompt

    if isinstance(examples, list):
        for example in examples:
            text = format_chat_prompt(example)
            texts.append(text)
    else:
        texts = format_chat_prompt(examples)

    return { "text" : texts, }


def formatting_prompts_func_technontech2claims_instruct_user_assistant(examples, tokenizer, eval_mode=False, force_output=False):
    system_prompt = instructions_dict['technontech2claims_instruct']
    # print('system_prompt', system_prompt)

    texts = []
    
    def format_chat_prompt(example):
        inputs_tech_abs       = example["technical_abstract"]
        inputs_non_tech_abs   = example.get("non_technical_abstract", '')
        inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
        
        # Assuming each example has a "messages" key which is a list of dicts like:
        # [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        messages = [
            {'role': 'user', 'content': system_prompt + '\n' + inputs + '\nPlease give me the VERIFIABLE CLAIMS in a json list alone without other information.'},
            # {'role': 'assistant', 'content': outputs}
        ]

        if not eval_mode:
            outputs               = f'''CLAIMS:```json\n{json.dumps(example['verifiable_claims'])}\n```''' if not eval_mode else ''
            messages.append({'role': 'assistant', 'content': outputs})
        else:
            if force_output:
                outputs               = f'''CLAIMS:```json'''
                messages.append({'role': 'assistant', 'content': outputs})
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # We create the string prompt first.
            add_generation_prompt=True  # This adds any necessary special tokens for generation.
        )
        # import pdb; pdb.set_trace()
        # Save the formatted prompt in a field (e.g., "text")
        return prompt

    if isinstance(examples, list):
        for example in examples:
            text = format_chat_prompt(example)
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            texts.append(text)
    else:
        texts = format_chat_prompt(examples)

    return { "text" : texts, }


### text -> claims

def formatting_prompts_func_text2claims_instruct(examples, tokenizer, eval_mode=False):
    system_prompt = instructions_dict['text2claims_instruct']
    
    texts = []
    
    def format_chat_prompt(example):
        inputs_tech_abs       = example["technical_abstract"]
        inputs_non_tech_abs   = example["non_technical_abstract"]
        inputs = 'TEXT:\n' + (inputs_tech_abs + ' ' + inputs_non_tech_abs).replace('\n', ' ') + '\n'
        # inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
        outputs               = f'''CLAIMS:```json\n{json.dumps(example['verifiable_claims'])}\n```'''

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': inputs},
            # {'role': 'assistant', 'content': outputs}
        ]

        if not eval_mode:
            messages.append({'role': 'assistant', 'content': outputs})
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # We create the string prompt first.
            add_generation_prompt=True  # This adds any necessary special tokens for generation.
        )
        return prompt

    if isinstance(examples, list):
        for example in examples:
            text = format_chat_prompt(example)
            texts.append(text)
    else:
        texts = format_chat_prompt(examples)

    return { "text" : texts, }

def formatting_prompts_func_text2claims_instruct_user_assistant(examples, tokenizer, eval_mode=False, force_output=False):
    system_prompt = instructions_dict['text2claims_instruct']
    # print('system_prompt', system_prompt)

    texts = []
    
    def format_chat_prompt(example):
        inputs_tech_abs       = example["technical_abstract"]
        inputs_non_tech_abs   = example.get("non_technical_abstract", '')
        # inputs = inputs_tech_abs + ' ' + inputs_non_tech_abs
        inputs = 'TEXT:\n' + (inputs_tech_abs + ' ' + inputs_non_tech_abs).replace('\n', ' ') + '\n'
        # inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
        
        # Assuming each example has a "messages" key which is a list of dicts like:
        # [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        messages = [
            {'role': 'user', 'content': system_prompt + '\n' + inputs + '\nPlease give me the VERIFIABLE CLAIMS in a json list alone without other information.'},
            # {'role': 'assistant', 'content': outputs}
        ]

        if not eval_mode:
            outputs               = f'''CLAIMS:```json\n{json.dumps(example['verifiable_claims'])}\n```''' if not eval_mode else ''
            messages.append({'role': 'assistant', 'content': outputs})
        else:
            if force_output:
                outputs               = f'''CLAIMS:```json'''
                messages.append({'role': 'assistant', 'content': outputs})
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # We create the string prompt first.
            add_generation_prompt=True  # This adds any necessary special tokens for generation.
        )
        # import pdb; pdb.set_trace()
        # Save the formatted prompt in a field (e.g., "text")
        return prompt

    if isinstance(examples, list):
        for example in examples:
            text = format_chat_prompt(example)
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            texts.append(text)
    else:
        texts = format_chat_prompt(examples)

    return { "text" : texts, }

### tech + non-tech -> ip

def formatting_prompts_func_technontech2ip(examples, tokenizer, eval_mode=False):
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

    def get_input_output(example):
        instructions = instructions_dict['technontech2ip']
        inputs_tech_abs       = example["technical_abstract"]
        inputs_non_tech_abs   = example["non_technical_abstract"]
        inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
        outputs               = f'''INVESTIGATION PROPOSALS:```json\n{json.dumps(example['investigation_proposals'])}\n```'''
        return instructions, inputs, outputs

    texts = []

    if isinstance(examples, list):
        for example in examples:
            instruction, input, output = get_input_output(example)
            if not eval_mode:
                text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            else:
                text = alpaca_prompt.format(instruction, input, '') + EOS_TOKEN
            # text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)

        # for instruction, input, output in zip(instructions, inputs, outputs):
        #     text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        #     texts.append(text)
    else:
        instructions, inputs, outputs = get_input_output(examples)
        if not eval_mode:
            texts = alpaca_prompt.format(instructions, inputs, outputs) + EOS_TOKEN
        else:
            texts = alpaca_prompt.format(instructions, inputs, '') + EOS_TOKEN
        # texts = alpaca_prompt.format(instructions, inputs, outputs) + EOS_TOKEN

    return { "text" : texts, }


def formatting_prompts_func_technontech2ip_instruct(examples, tokenizer, eval_mode=False):
    system_prompt = instructions_dict['technontech2ip_instruct']
    
    texts = []
    
    def format_chat_prompt(example):
        inputs_tech_abs       = example["technical_abstract"]
        inputs_non_tech_abs   = example["non_technical_abstract"]
        inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
        outputs               = f'''INVESTIGATION PROPOSALS:```json\n{json.dumps(example['investigation_proposals'])}\n```'''

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': inputs},
            # {'role': 'assistant', 'content': outputs}
        ]
        if not eval_mode:
            messages.append({'role': 'assistant', 'content': outputs})
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # We create the string prompt first.
            add_generation_prompt=True  # This adds any necessary special tokens for generation.
        )
        return prompt

    if isinstance(examples, list):
        for example in examples:
            text = format_chat_prompt(example)
            texts.append(text)
    else:
        texts = format_chat_prompt(examples)

    return { "text" : texts, }

def formatting_prompts_func_technontech2ip_instruct_user_assistant(examples, tokenizer, eval_mode=False, force_output=False):
    system_prompt = instructions_dict['technontech2ip_instruct']
    # print('system_prompt', system_prompt)

    texts = []
    
    def format_chat_prompt(example):
        inputs_tech_abs       = example["technical_abstract"]
        inputs_non_tech_abs   = example.get("non_technical_abstract", '')
        inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
        # outputs               = f'''INVESTIGATION PROPOSALS:```json\n{json.dumps(example['investigation_proposals'])}\n```'''
        
        # Assuming each example has a "messages" key which is a list of dicts like:
        # [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        messages = [
            {'role': 'user', 'content': system_prompt + '\n' + inputs + '\nPlease give me the INVESTIGATION PROPOSALS in a json list alone without other information.'},
        ]
        # if not eval_mode:
        #     messages.append({'role': 'assistant', 'content': outputs})
        if not eval_mode:
            outputs               = f'''INVESTIGATION PROPOSALS:```json\n{json.dumps(example['investigation_proposals'])}\n```''' if not eval_mode else ''
            messages.append({'role': 'assistant', 'content': outputs})
        else:
            if force_output:
                outputs               = f'''INVESTIGATION PROPOSALS:```json'''
                messages.append({'role': 'assistant', 'content': outputs})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # We create the string prompt first.
            add_generation_prompt=True  # This adds any necessary special tokens for generation.
        )
        # Save the formatted prompt in a field (e.g., "text")
        return prompt

    if isinstance(examples, list):
        for example in examples:
            text = format_chat_prompt(example)
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            texts.append(text)
    else:
        texts = format_chat_prompt(examples)

    return { "text" : texts, }

### text -> ip

def formatting_prompts_func_text2ip_instruct(examples, tokenizer, eval_mode=False):
    system_prompt = instructions_dict['text2ip_instruct']
    
    texts = []
    
    def format_chat_prompt(example):
        inputs_tech_abs       = example["technical_abstract"]
        inputs_non_tech_abs   = example["non_technical_abstract"]
        # inputs = 'TEXT:\n' + inputs_tech_abs + ' ' + inputs_non_tech_abs
        inputs = 'TEXT:\n' + (inputs_tech_abs + ' ' + inputs_non_tech_abs).replace('\n', ' ') + '\n'
        # inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
        outputs               = f'''INVESTIGATION PROPOSALS:```json\n{json.dumps(example['investigation_proposals'])}\n```'''

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': inputs},
            # {'role': 'assistant', 'content': outputs}
        ]
        if not eval_mode:
            messages.append({'role': 'assistant', 'content': outputs})
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # We create the string prompt first.
            add_generation_prompt=True  # This adds any necessary special tokens for generation.
        )
        return prompt

    if isinstance(examples, list):
        for example in examples:
            text = format_chat_prompt(example)
            texts.append(text)
    else:
        texts = format_chat_prompt(examples)

    return { "text" : texts, }

def formatting_prompts_func_text2ip_instruct_user_assistant(examples, tokenizer, eval_mode=False, force_output=False):
    system_prompt = instructions_dict['text2ip_instruct']
    # print('system_prompt', system_prompt)

    texts = []
    
    def format_chat_prompt(example):
        inputs_tech_abs       = example["technical_abstract"]
        inputs_non_tech_abs   = example.get("non_technical_abstract", '')
        # inputs = inputs_tech_abs + ' ' + inputs_non_tech_abs
        inputs = 'TEXT:\n' + (inputs_tech_abs + ' ' + inputs_non_tech_abs).replace('\n', ' ') + '\n'
        # inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
        # outputs               = f'''INVESTIGATION PROPOSALS:```json\n{json.dumps(example['investigation_proposals'])}\n```'''
        
        # Assuming each example has a "messages" key which is a list of dicts like:
        # [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        messages = [
            {'role': 'user', 'content': system_prompt + '\n' + inputs + '\nPlease give me the INVESTIGATION PROPOSALS in a json list alone without other information.'},
        ]
        # if not eval_mode:
        #     messages.append({'role': 'assistant', 'content': outputs})
        if not eval_mode:
            outputs               = f'''INVESTIGATION PROPOSALS:```json\n{json.dumps(example['investigation_proposals'])}\n```''' if not eval_mode else ''
            messages.append({'role': 'assistant', 'content': outputs})
        else:
            if force_output:
                outputs               = f'''INVESTIGATION PROPOSALS:```json'''
                messages.append({'role': 'assistant', 'content': outputs})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # We create the string prompt first.
            add_generation_prompt=True  # This adds any necessary special tokens for generation.
        )
        # Save the formatted prompt in a field (e.g., "text")
        return prompt

    if isinstance(examples, list):
        for example in examples:
            text = format_chat_prompt(example)
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            texts.append(text)
    else:
        texts = format_chat_prompt(examples)

    return { "text" : texts, }

### tech + non-tech -> claims + ip

def formatting_prompts_func_technontech2claimsip(examples, tokenizer, eval_mode=False):
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

    def get_input_output(example):
        instructions = instructions_dict['technontech2claimsip']
        inputs_tech_abs       = example["technical_abstract"]
        inputs_non_tech_abs   = example["non_technical_abstract"]
        inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
        outputs               = f'''```json\n{{"claims": {json.dumps(example['verifiable_claims'])},\n"investigation_proposals": {json.dumps(example['investigation_proposals'])}}}\n```'''
        return instructions, inputs, outputs
        
    texts = []

    if isinstance(examples, list):
        for example in examples:
            instruction, input, output = get_input_output(example)
            if not eval_mode:
                text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            else:
                text = alpaca_prompt.format(instruction, input, '') + EOS_TOKEN
            # text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)

    else:
        instructions, inputs, outputs = get_input_output(examples)
        if not eval_mode:
            texts = alpaca_prompt.format(instructions, inputs, outputs) + EOS_TOKEN
        else:
            texts = alpaca_prompt.format(instructions, inputs, '') + EOS_TOKEN
        # texts = alpaca_prompt.format(instructions, inputs, outputs) + EOS_TOKEN

    return { "text" : texts, }


def formatting_prompts_func_technontech2claimsip_instruct(examples, tokenizer, eval_mode=False):
    system_prompt = instructions_dict['technontech2claimsip_instruct']
    
    texts = []
    
    def format_chat_prompt(example):
        inputs_tech_abs       = example["technical_abstract"]
        inputs_non_tech_abs   = example["non_technical_abstract"]
        inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
        outputs               = f'''```json\n{{"claims": {json.dumps(example['verifiable_claims'])},\n"investigation_proposals": {json.dumps(example['investigation_proposals'])}}}\n```'''

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': inputs},
            # {'role': 'assistant', 'content': outputs}
        ]
        if not eval_mode:
            messages.append({'role': 'assistant', 'content': outputs})
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # We create the string prompt first.
            add_generation_prompt=True  # This adds any necessary special tokens for generation.
        )
        return prompt

    if isinstance(examples, list):
        for example in examples:
            text = format_chat_prompt(example)
            texts.append(text)
    else:
        texts = format_chat_prompt(examples)

    return { "text" : texts, }


def formatting_prompts_func_technontech2claimsip_instruct_user_assistant(examples, tokenizer, eval_mode=False):
    system_prompt = instructions_dict['technontech2claimsip_instruct']
    # print('system_prompt', system_prompt)

    texts = []
    
    def format_chat_prompt(example):
        inputs_tech_abs       = example["technical_abstract"]
        inputs_non_tech_abs   = example["non_technical_abstract"]
        inputs                = 'TECHNICAL ABSTRACT:\n' + inputs_tech_abs + '\nNON-TECHNICAL ABSTRACT:\n' + inputs_non_tech_abs
        outputs               = f'''```json\n{{"claims": {json.dumps(example['verifiable_claims'])},\n"investigation_proposals": {json.dumps(example['investigation_proposals'])}}}\n```'''
        
        # Assuming each example has a "messages" key which is a list of dicts like:
        # [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        messages = [
            {'role': 'user', 'content': system_prompt + '\n' + inputs + '\nPlease give me the INVESTIGATION PROPOSALS in a json list alone without other information.'},
        ]
        if not eval_mode:
            messages.append({'role': 'assistant', 'content': outputs})
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,             # We create the string prompt first.
            add_generation_prompt=True  # This adds any necessary special tokens for generation.
        )
        # Save the formatted prompt in a field (e.g., "text")
        return prompt

    if isinstance(examples, list):
        for example in examples:
            text = format_chat_prompt(example)
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            texts.append(text)
    else:
        texts = format_chat_prompt(examples)

    return { "text" : texts, }

format_prompts_func_dict = {
    'tech2nontech': formatting_prompts_func_tech2nontech,
    'tech2nontech_instruct': formatting_prompts_func_tech2nontech_instruct,
    'tech2nontech_instruct_user_assistant': formatting_prompts_func_tech2nontech_instruct_user_assistant,
    'technontech2claims': formatting_prompts_func_technontech2claims,
    'technontech2claims_instruct': formatting_prompts_func_technontech2claims_instruct,
    'technontech2claims_instruct_user_assistant': formatting_prompts_func_technontech2claims_instruct_user_assistant,
    'text2claims_instruct': formatting_prompts_func_text2claims_instruct,
    'text2claims_instruct_user_assistant': formatting_prompts_func_text2claims_instruct_user_assistant,
    'technontech2ip': formatting_prompts_func_technontech2ip,
    'technontech2ip_instruct': formatting_prompts_func_technontech2ip_instruct,
    'technontech2ip_instruct_user_assistant': formatting_prompts_func_technontech2ip_instruct_user_assistant,
    'text2ip_instruct': formatting_prompts_func_text2ip_instruct,
    'text2ip_instruct_user_assistant': formatting_prompts_func_text2ip_instruct_user_assistant,
    'technontech2claimsip': formatting_prompts_func_technontech2claimsip,
    'technontech2claimsip_instruct': formatting_prompts_func_technontech2claimsip_instruct,
    'technontech2claimsip_instruct_user_assistant': formatting_prompts_func_technontech2claimsip_instruct_user_assistant,
}

def apply_formatting_prompt(formatting_prompts_func, example, tokenizer, eval_mode=False, force_output=False):
    kwargs = {"tokenizer": tokenizer, "eval_mode": eval_mode}
    if force_output and "force_output" in inspect.signature(formatting_prompts_func).parameters:
        kwargs["force_output"] = force_output
    return formatting_prompts_func(example, **kwargs)

def get_nsf_data_proc(ROOT_DIR, tokenizer, mode='tech2nontech', formatting_prompts_func=None, eval_mode=False):
    train_dataset, val_dataset, test_dataset = get_nsf_data_raw(ROOT_DIR)
    
    if formatting_prompts_func is None:
        formatting_prompts_func = format_prompts_func_dict[mode]

    func = lambda x: formatting_prompts_func(x, tokenizer=tokenizer, eval_mode=eval_mode)
    
    train_dataset = train_dataset.map(func, batched = False,)
    val_dataset = val_dataset.map(func, batched = False,)
    test_dataset = test_dataset.map(func, batched = False,)
    return train_dataset, val_dataset, test_dataset

def get_nsf_data_all_proc(ROOT_DIR, tokenizer, mode='tech2nontech', formatting_prompts_func=None, eval_mode=False, force_output=False):
    all_dataset = get_nsf_data_all_raw(ROOT_DIR)

    all_dataset = all_dataset.rename_column("abstract", "technical_abstract") #.map(lambda x: {"non_technical_abstract": x["technical_abstract"]})

    if formatting_prompts_func is None:
        formatting_prompts_func = format_prompts_func_dict[mode]

    func = lambda x: apply_formatting_prompt(formatting_prompts_func, x, tokenizer=tokenizer, eval_mode=eval_mode, force_output=force_output)
    
    all_dataset = all_dataset.map(func, batched = False,)
    return all_dataset

def get_nsf_data_20k_proc(ROOT_DIR, tokenizer, mode='tech2nontech', formatting_prompts_func=None, eval_mode=False, force_output=False):
    train_dataset, val_dataset, test_dataset = get_nsf_data_20k_raw(ROOT_DIR)
    
    if formatting_prompts_func is None:
        formatting_prompts_func = format_prompts_func_dict[mode]

    if 'tech2nontech' in mode:
        # remove entries with empty technical_abstract or non_technical_abstract
        train_dataset = train_dataset.filter(lambda x: x['technical_abstract'] not in ['', None] and x['non_technical_abstract'] not in ['', None])
        val_dataset = val_dataset.filter(lambda x: x['technical_abstract'] not in ['', None] and x['non_technical_abstract'] not in ['', None])
        test_dataset = test_dataset.filter(lambda x: x['technical_abstract'] not in ['', None] and x['non_technical_abstract'] not in ['', None])

    func = lambda x: apply_formatting_prompt(formatting_prompts_func, x, tokenizer=tokenizer, eval_mode=eval_mode, force_output=force_output)

    train_dataset = train_dataset.map(func, batched = False,)
    val_dataset = val_dataset.map(func, batched = False,)
    test_dataset = test_dataset.map(func, batched = False,)
    return train_dataset, val_dataset, test_dataset



def cast_award_id_to_string(ds: Dataset) -> Dataset:
    # this will turn award_id into a string feature
    return ds.cast_column("award_id", Value("string"))


# train_dataset   = cast_award_id_to_string(train_dataset)
# train_dataset_20k = cast_award_id_to_string(train_dataset_20k)

# # now award_id is string in both → won't be dropped
# combined = concat_datasets_drop_mismatches([train_dataset, train_dataset_20k])


def get_nsf_data_matsci_and_20k_raw(ROOT_DIR):
    # matsci_dataset = get_nsf_data_raw(ROOT_DIR)
    train_dataset, val_dataset, test_dataset = get_nsf_data_raw(ROOT_DIR)
    # 20k_dataset = get_nsf_data_20k_raw(ROOT_DIR)
    train_dataset_20k, val_dataset_20k, test_dataset_20k = get_nsf_data_20k_raw(ROOT_DIR)

    train_dataset = cast_award_id_to_string(train_dataset)
    train_dataset_20k = cast_award_id_to_string(train_dataset_20k)
    val_dataset = cast_award_id_to_string(val_dataset)
    val_dataset_20k = cast_award_id_to_string(val_dataset_20k)
    test_dataset = cast_award_id_to_string(test_dataset)
    test_dataset_20k = cast_award_id_to_string(test_dataset_20k)

    train_dataset = concat_datasets_drop_mismatches([train_dataset, train_dataset_20k])
    val_dataset = concat_datasets_drop_mismatches([val_dataset, val_dataset_20k])
    test_dataset = concat_datasets_drop_mismatches([test_dataset, test_dataset_20k])

    return train_dataset, val_dataset, test_dataset

def get_nsf_data_matsci_and_20k_proc(ROOT_DIR, tokenizer, mode='tech2nontech', formatting_prompts_func=None, eval_mode=False):
    train_dataset, val_dataset, test_dataset = get_nsf_data_matsci_and_20k_raw(ROOT_DIR)

    if formatting_prompts_func is None:
        formatting_prompts_func = format_prompts_func_dict[mode]

    if 'tech2nontech' in mode:
        # remove entries with empty technical_abstract or non_technical_abstract
        train_dataset = train_dataset.filter(lambda x: x['technical_abstract'] not in ['', None] and x['non_technical_abstract'] not in ['', None])
        val_dataset = val_dataset.filter(lambda x: x['technical_abstract'] not in ['', None] and x['non_technical_abstract'] not in ['', None])
        test_dataset = test_dataset.filter(lambda x: x['technical_abstract'] not in ['', None] and x['non_technical_abstract'] not in ['', None])

    func = lambda x: formatting_prompts_func(x, tokenizer=tokenizer, eval_mode=eval_mode)

    train_dataset = train_dataset.map(func, batched = False,)
    val_dataset = val_dataset.map(func, batched = False,)
    test_dataset = test_dataset.map(func, batched = False,)
    return train_dataset, val_dataset, test_dataset
