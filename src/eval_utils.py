import evaluate
from tqdm.auto import tqdm
from data_utils import alpaca_prompt, instructions_dict
from text_utils import text2json
from claims_utils import compare_claims
import os
import json
import numpy as np
import math
import time
from torch.utils.data import DataLoader
import sacrebleu
import evaluate
from collections import defaultdict


# class Metrics():
#     def __init__(self):
#         self.bertscore = evaluate.load("bertscore")
#         self.rouge = evaluate.load("rouge")
#         self.bleu = evaluate.load("bleu")
        
#     def scores(self, predictions: list, references: list, verbose: bool = False):
#         if verbose:
#             print('Computing BERTScore...')
#         bertscore_results = self.bertscore.compute(predictions=predictions, references=references, lang="en")
#         if verbose:
#             print('Computing ROUGE...')
#         rouge_results = self.rouge.compute(predictions=predictions, references=references)
#         if verbose:
#             print('Computing BLEU...')
#         bleu_results = self.bleu.compute(predictions=predictions, references=references)
        
#         return {
#             'bertscore': bertscore_results,
#             'rouge': rouge_results,
#             'bleu': bleu_results
#         }
    
#     @staticmethod
#     def concise(scores_results):
#         return {
#             'bertscore': {key: scores_results['bertscore'][key] for key in ['precision', 'recall', 'f1']},
#             'rouge': scores_results['rouge'],
#             'bleu': {key: scores_results['bleu'][key] for key in ['bleu']}
#         }


class Metrics:
    def __init__(self):
        self.bertscore = evaluate.load("bertscore")
        self.rouge     = evaluate.load("rouge")
        self.bleu      = evaluate.load("bleu")

    @staticmethod
    def _dict_of_lists(d):
        return {k: list(v) for k, v in d.items()}

    def _sentence_bleu_list(self, preds, refs):
        """Compute BLEU for each (pred, ref) pair with SacreBLEU."""
        return [
            sacrebleu.sentence_bleu(p, [r]).score
            for p, r in zip(preds, refs)
        ]

    # ------------------------------------------------------------------
    def scores(self, predictions, references, verbose=False):
        if verbose: print("Computing BERTScore …")
        bs = self.bertscore.compute(
            predictions=predictions, references=references, lang="en"
        )

        if verbose: print("Computing ROUGE …")
        rouge_raw = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=False,
        )
        rouge_lists = self._dict_of_lists(rouge_raw)

        if verbose: print("Computing BLEU …")
        try:
            bleu_raw = self.bleu.compute(
                predictions=predictions,
                references=references,
                return_all_scores=True,     # works on evaluate ≥ 0.4.0
            )
            bleu_lists = {"bleu": bleu_raw["bleu"]}
        except TypeError:
            # Older evaluate version – fall back to sentence loop
            bleu_lists = {"bleu": self._sentence_bleu_list(predictions, references)}

        return {"bertscore": bs, "rouge": rouge_lists, "bleu": bleu_lists}

    # concise() unchanged
    @staticmethod
    def concise(scores_results):
        return {
            "bertscore": {k: scores_results["bertscore"][k]
                          for k in ["precision", "recall", "f1"]},
            "rouge":     scores_results["rouge"],
            "bleu":      scores_results["bleu"],
        }

        if verbose: print("Computing BERTScore …")
        bs = self.bertscore.compute(
            predictions=predictions, references=references, lang="en"
        )                                  # lists already

        if verbose: print("Computing ROUGE …")
        rouge_raw = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=False,
        )
        rouge_lists = {k: list(v) for k, v in rouge_raw.items()}

        if verbose: print("Computing BLEU …")
        try:
            bleu_raw = self.bleu.compute(
                predictions=predictions,
                references=references,
                return_all_scores=True,     # works on evaluate ≥ 0.4.0
            )
            bleu_lists = {"bleu": bleu_raw["bleu"]}
        except TypeError:
            # Older evaluate version – fall back to sentence loop
            bleu_lists = {"bleu": self.bleu._sentence_bleu_list(predictions, references)}

        return {"bertscore": bs, "rouge": rouge_lists, "bleu": bleu_lists}


    # ───────────────────────────────────────────────────────────
    # Keep the concise helper unchanged (it already expects lists)
    # ───────────────────────────────────────────────────────────
    @staticmethod
    def concise(scores_results):
        return {
            "bertscore": {k: scores_results["bertscore"][k]
                          for k in ["precision", "recall", "f1"]},
            "rouge":     scores_results["rouge"],
            "bleu":      scores_results["bleu"],
        }



def evaluate_model(model, tokenizer, test_dataset, debug=False):
    
    metrics = Metrics()

    results = []
    technical_abstracts = []
    non_technical_abstracts = []
    preds = []
    for i in tqdm(range(len(test_dataset))):
        if debug and i % 2 != 0:
            continue
        inputs = tokenizer(test_dataset[i]['text'], return_tensors = "pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens = 2000, use_cache = True)
        # import pdb; pdb.set_trace()
        if 'instruct' in model.config._name_or_path.lower():
            pred = tokenizer.decode(outputs[0]).replace(test_dataset[i]['text'], '').strip()
        else:
            pred = tokenizer.batch_decode(outputs)[0].split('### Response:')[1].strip()
        metrics_scores = metrics.scores([pred], [test_dataset[i]['non_technical_abstract']])
        concise_scores = metrics.concise(metrics_scores)
        results.append(concise_scores)
        preds.append(pred)
        technical_abstracts.append(test_dataset[i]['technical_abstract'])
        non_technical_abstracts.append(test_dataset[i]['non_technical_abstract'])

    return results, preds, technical_abstracts, non_technical_abstracts


def generate_predictions_batch(model, tokenizer, test_dataset, batch_size=8, debug=False):
    def collate_fn(batch):
        # batch is a list of dicts, each with a "text" field
        texts = [ex["text"] for ex in batch]
        # tokenizer will pad to the longest in the batch
        tokenized = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return tokenized, batch  # Return both tokenized data and original batch

    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_outputs = []
    all_inputs = []
    start = time.time()

    for i, (batch_tokenized, original_batch) in tqdm(enumerate(dataloader), total=len(dataloader)):
        if debug and i > 2:
            continue
        # move to GPU
        batch_tokenized = {k: v.to("cuda") for k, v in batch_tokenized.items()}
        # generate
        out_ids = model.generate(
            **batch_tokenized,
            max_new_tokens=1000,
            use_cache=True,
            do_sample=False
        )
        # decode to strings
        decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        all_outputs.extend(decoded)
        all_inputs.extend(original_batch)  # Store the original examples

    return all_inputs, all_outputs
    

# def evaluate_model_claims(model, tokenizer, test_dataset, debug=False, save_dir=None, gen_only=False):
    
#     system_prompt = "Check two scientific claims c1 and c2, if c1 is supported by c2. If c2 includes all the evidences for c1, but also includes additional content, then it should still be supported (YES). If not all information of c1 is included in c2, or if c2 contains information that conflicts with information in c1, then it should be unsupported (NO). Answer only as a YES or NO."
    
#     results_whole = []
#     results = []
#     technical_abstracts = []
#     non_technical_abstracts = []
#     gold_claims_all = []
#     pred_claims_all = []
#     preds = []
#     gen_only_results = []

#     for i in tqdm(range(len(test_dataset))):
#         preds_only_path = os.path.join(save_dir, f"{i}_preds_gen_only.json")
#         if os.path.exists(preds_only_path):
#             continue
    
#         if debug and i % 100 != 0:
#             continue

#         save_path = None
#         if save_dir is not None:
#             save_path = os.path.join(save_dir, f"{i}.json")
#             preds_save_path = os.path.join(save_dir, f"{i}_preds.json")
#             if os.path.exists(save_path) and os.path.exists(preds_save_path):
#                 scores = json.load(open(save_path))
#                 concise_scores = {
#                     'precision': scores['precision'],
#                     'recall': scores['recall'],
#                     'fscore': scores['fscore']
#                 }
#                 results.append(concise_scores)
#                 pred_dict = json.load(open(preds_save_path))
#                 results.append(pred_dict['results_whole'])
#                 preds.append(pred_dict['preds'])
#                 technical_abstracts.append(pred_dict['technical_abstracts'])
#                 non_technical_abstracts.append(pred_dict['non_technical_abstracts'])
#                 gold_claims_all.append(pred_dict['gold_claims_all'])
#                 pred_claims_all.append(pred_dict['pred_claims_all'])

#                 continue

#         inputs = tokenizer(test_dataset[i]['text'], return_tensors = "pt").to("cuda")

#         outputs = model.generate(**inputs, max_new_tokens = 500, use_cache = True)
        
#         if 'instruct' in model.config._name_or_path.lower():
#             pred = tokenizer.decode(outputs[0]).replace(test_dataset[i]['text'], '').strip()
#         else:
#             pred = tokenizer.batch_decode(outputs)[0].split('### Response:')[1].strip()

#         # inputs1 = tokenizer([test_dataset[i]['text'], test_dataset[i+1]['text']], return_tensors = "pt", padding=True).to("cuda")

#         # outputs1 = model.generate(**inputs1, max_new_tokens = 500, use_cache = True)
#         # # import pdb; pdb.set_trace()
#         # if 'instruct' in model.config._name_or_path.lower():
#         #     pred1 = tokenizer.decode(outputs1[0]).replace(test_dataset[i]['text'], '').strip()
#         # else:
#         #     pred1 = tokenizer.batch_decode(outputs)[0].split('### Response:')[1].strip()

#         # inputs2 = tokenizer([test_dataset[i]['text'], test_dataset[i]['text']], return_tensors = "pt", padding=True).to("cuda")

#         # outputs2 = model.generate(**inputs2, max_new_tokens = 500, use_cache = True)
#         # # import pdb; pdb.set_trace()
#         # if 'instruct' in model.config._name_or_path.lower():
#         #     pred2 = tokenizer.decode(outputs2[0]).replace(test_dataset[i]['text'], '').strip()
#         # else:
#         #     pred2 = tokenizer.batch_decode(outputs)[0].split('### Response:')[1].strip()
#         # import pdb; pdb.set_trace()

#         if gen_only:
#             # preds.append(pred)
#             gen_only_results.append({
#                 'pred': pred,
#                 'technical_abstract': test_dataset[i]['technical_abstract'],
#                 'non_technical_abstract': test_dataset[i]['non_technical_abstract'],
#                 'verifiable_claims': test_dataset[i]['verifiable_claims'] if 'verifiable_claims' in test_dataset[i] else None
#             })

#             if save_dir is not None:
#                 os.makedirs(save_dir, exist_ok=True)
#                 with open(preds_only_path, "w") as f:
#                     json.dump(gen_only_results[-1], f, indent=4)
#             continue

#         try:   
#             pred_claims = text2json(pred)
#         except:
#             import pdb; pdb.set_trace()
#             pred_claims = text2json(pred)
#         gold_claims = test_dataset[i]['verifiable_claims']
        
#         try:
#             scores = compare_claims(pred_claims, gold_claims, return_all=True, system_prompt=system_prompt, threshold=0.9)
#         except:
#             import pdb; pdb.set_trace()
#             scores = compare_claims(pred_claims, gold_claims, return_all=True, system_prompt=system_prompt, threshold=0.9)
#         concise_scores = {
#             'precision': scores['precision'],
#             'recall': scores['recall'],
#             'fscore': scores['fscore']
#         }

#         results_whole.append(scores)
#         results.append(concise_scores)
#         preds.append(pred)
#         technical_abstracts.append(test_dataset[i]['technical_abstract'])
#         non_technical_abstracts.append(test_dataset[i]['non_technical_abstract'])
#         gold_claims_all.append(gold_claims)
#         pred_claims_all.append(pred_claims)

#         if save_path is not None:
#             os.makedirs(save_dir, exist_ok=True)
#             with open(save_path, "w") as f:
#                 json.dump(scores, f, indent=4)

#             pred_dict = {
#                 "results_whole": scores, 
#                 "preds": pred, "technical_abstracts": test_dataset[i]['technical_abstract'],
#                 "non_technical_abstracts": test_dataset[i]['non_technical_abstract'],
#                 "gold_claims_all": gold_claims, "pred_claims_all": pred_claims}
#             with open(os.path.join(save_dir, f"{i}_preds.json"), "w") as f:
#                 json.dump(pred_dict, f, indent=4)

#     if gen_only:
#         return gen_only_results
#     else:
#         return results, results_whole, preds, technical_abstracts, non_technical_abstracts, gold_claims_all, pred_claims_all


def evaluate_model_claims(model, tokenizer, test_dataset, batch_size=8, debug=False, save_dir=None, gen_only=False, force_output=False):
    """
    Evaluate model claims using a DataLoader to generate predictions in batches.

    Args:
        model: The language model.
        tokenizer: The tokenizer associated with the model.
        test_dataset: A list (or similar) of examples. Each example should have at least:
            - 'text'
            - 'technical_abstract'
            - 'non_technical_abstract'
            - 'verifiable_claims' (if available)
        batch_size (int): Number of examples to process per batch.
        debug (bool): If True, only process examples with indices that are multiples of 100.
        save_dir (str): Directory in which to save per-example outputs.
        gen_only (bool): If True, only generate predictions without evaluation.

    Returns:
        If gen_only is True: a list of dictionaries with generated predictions and associated fields.
        Otherwise: a tuple of lists containing evaluation results and related fields.
    """
    

    import os
    import json
    import torch
    from torch.utils.data import Dataset, DataLoader, Subset
    from tqdm import tqdm

    # Define a simple Dataset wrapper that returns (index, sample)
    class IndexedDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return idx, self.dataset[idx]

    # Custom collate function that simply unzips the batch into indices and samples
    def collate_fn(batch):
        indices, samples = zip(*batch)
        return indices, samples

    # Wrap the test_dataset and filter based on the debug flag.
    indexed_dataset = IndexedDataset(test_dataset)
    if debug:
        debug_indices = [i for i in range(len(test_dataset)) if i % 20 == 0]
        indexed_dataset = Subset(indexed_dataset, debug_indices)

    dataloader = DataLoader(indexed_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    system_prompt = ("Check two scientific claims c1 and c2, if c1 is supported by c2. "
                     "If c2 includes all the evidences for c1, but also includes additional content, then it should still be supported (YES). "
                     "If not all information of c1 is included in c2, or if c2 contains information that conflicts with information in c1, then it should be unsupported (NO). "
                     "Answer only as a YES or NO.")

    results_whole = []
    results = []
    technical_abstracts = []
    non_technical_abstracts = []
    gold_claims_all = []
    pred_claims_all = []
    preds = []
    gen_only_results = []

    # Process batches from the DataLoader
    for batch_indices, batch_samples in tqdm(dataloader):
        batch_preds_save_path = os.path.join(save_dir, f"batch_{batch_indices[0]}-{batch_indices[-1]}_preds_gen_only.json")
        # if os.path.exists(batch_preds_save_path):
        #     continue
        indices_to_generate = []  # List of indices within the batch that need processing
        samples_to_generate = []  # Corresponding samples for which generation is required

        # Check each sample for existing saved output.
        for i, sample in zip(batch_indices, batch_samples):
            if save_dir is not None:
                save_path = os.path.join(save_dir, f"{i}.json")
                preds_save_path = os.path.join(save_dir, f"{i}_preds.json")
                if os.path.exists(save_path) and os.path.exists(preds_save_path):
                    scores = json.load(open(save_path))
                    concise_scores = {
                        'precision': scores['precision'],
                        'recall': scores['recall'],
                        'fscore': scores['fscore']
                    }
                    results.append(concise_scores)
                    pred_dict = json.load(open(preds_save_path))
                    results_whole.append(pred_dict['results_whole'])
                    preds.append(pred_dict['preds'])
                    technical_abstracts.append(pred_dict['technical_abstracts'])
                    non_technical_abstracts.append(pred_dict['non_technical_abstracts'])
                    gold_claims_all.append(pred_dict['gold_claims_all'])
                    pred_claims_all.append(pred_dict['pred_claims_all'])
                    continue  # Skip generation for this sample if saved.
            indices_to_generate.append(i)
            samples_to_generate.append(sample)

        # If there are samples to generate in this batch, process them together.
        if samples_to_generate:
            # import pdb; pdb.set_trace()
            texts = [sample['text'] for sample in samples_to_generate]
            tokenizer.padding_side = 'left'
            if tokenizer.padding_side != 'left':
                print('tokenizer.padding_side', tokenizer.padding_side)
            # print('tokenizer.pad_token', tokenizer.pad_token)
            tokenizer.pad_token = tokenizer.eos_token
            inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True).to("cuda")
            if force_output:
                inputs = {key: val[:,:-1] for key, val in inputs.items()}

                                # add_special_tokens=False
                                # truncation=True
                                # ).to("cuda")
            
            outputs = model.generate(**inputs, max_new_tokens=500) #, use_cache=True)
            batch_predictions = tokenizer.batch_decode(outputs)
            # import pdb; pdb.set_trace()

            # Process each generated prediction.
            for j, i in enumerate(indices_to_generate):
                pred_raw = batch_predictions[j]
                # Process the prediction based on model type.
                if 'instruct' in model.config._name_or_path.lower():
                    if force_output:
                        pred = pred_raw.replace(samples_to_generate[j]['text'][3:-4], '').strip()
                    else:
                        pred = pred_raw.replace(samples_to_generate[j]['text'], '').strip()
                    # pred = pred_raw.replace(samples_to_generate[j]['text'], '').strip()
                else:
                    try:
                        pred = pred_raw.split('### Response:')[1].strip()
                    except IndexError:
                        pred = pred_raw.strip()
                # print(pred)
                # import pdb; pdb.set_trace()

                if gen_only:
                    gen_only_results.append({
                        'pred': pred,
                        'technical_abstract': samples_to_generate[j]['technical_abstract'],
                        'non_technical_abstract': samples_to_generate[j].get('non_technical_abstract', None),
                        'verifiable_claims': samples_to_generate[j].get('verifiable_claims', None),
                        'award_id': samples_to_generate[j].get('award_id', None)
                    })

                    
                else:
                    try:
                        pred_claims = text2json(pred)
                    except Exception:
                        import pdb; pdb.set_trace()
                        pred_claims = text2json(pred)
                    gold_claims = samples_to_generate[j]['verifiable_claims']

                    try:
                        scores = compare_claims(pred_claims, gold_claims, return_all=True,
                                                system_prompt=system_prompt, threshold=0.9)
                    except Exception:
                        import pdb; pdb.set_trace()
                        scores = compare_claims(pred_claims, gold_claims, return_all=True,
                                                system_prompt=system_prompt, threshold=0.9)
                    concise_scores = {
                        'precision': scores['precision'],
                        'recall': scores['recall'],
                        'fscore': scores['fscore']
                    }

                    results_whole.append(scores)
                    results.append(concise_scores)
                    preds.append(pred)
                    technical_abstracts.append(samples_to_generate[j]['technical_abstract'])
                    non_technical_abstracts.append(samples_to_generate[j]['non_technical_abstract'])
                    gold_claims_all.append(gold_claims)
                    pred_claims_all.append(pred_claims)

                    # Save outputs for this sample if a save directory is provided.
                    if save_dir is not None:
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{i}.json")
                        preds_save_path = os.path.join(save_dir, f"{i}_preds.json")
                        with open(save_path, "w") as f:
                            json.dump(scores, f, indent=4)
                        pred_dict = {
                            "results_whole": scores,
                            "preds": pred,
                            "technical_abstracts": samples_to_generate[j]['technical_abstract'],
                            "non_technical_abstracts": samples_to_generate[j]['non_technical_abstract'],
                            "gold_claims_all": gold_claims,
                            "pred_claims_all": pred_claims
                        }
                        with open(preds_save_path, "w") as f:
                            json.dump(pred_dict, f, indent=4)

            if gen_only and save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                
                pred_dict = {
                    'batch_predictions': batch_predictions,
                    'samples_to_generate': samples_to_generate
                    # "preds": [batch_predictions],
                    # "technical_abstracts": samples_to_generate['technical_abstract'],
                    # "non_technical_abstracts": samples_to_generate['non_technical_abstract'],
                    # "gold_claims_all": [samples_to_generate[j].get('verifiable_claims', None) for j in indices_to_generate],
                    # "pred_claims_all": [text2json(pred) for j in indices_to_generate]
                }
                with open(batch_preds_save_path, "w") as f:
                    json.dump(pred_dict, f, indent=4)

                # os.makedirs(save_dir, exist_ok=True)
                # with open(os.path.join(save_dir, 'gen_only_results.jsonl'), 'w') as f:
                #     for item in gen_only_results:
                #         f.write(json.dumps(item) + '\n')
                # print(f"Saved gen_only_results to: {os.path.join(save_dir, 'gen_only_results.jsonl')}

        # break

    if gen_only:
        return gen_only_results
    else:
        return (results, results_whole, preds, technical_abstracts,
                non_technical_abstracts, gold_claims_all, pred_claims_all)



# def evaluate_model_ips(model, tokenizer, test_dataset, debug=False, save_dir=None, gen_only=False):
    
#     # system_prompt = "Check two scientific claims c1 and c2, if c1 is supported by c2. If c2 includes all the evidences for c1, but also includes additional content, then it should still be supported (YES). If not all information of c1 is included in c2, or if c2 contains information that conflicts with information in c1, then it should be unsupported (NO). Answer only as a YES or NO."
#     system_prompt = "Check two investigation proposals c1 and c2, if c1 is supported by c2. If c2 includes all the investigations proposed by c1, but also includes additional proposals, then it should still be supported (YES). If not all proposed investigations by c1 is included in c2, or if c2 contains investigation actions that conflict with investigation actions in c1, then it should be unsupported (NO). Answer only as a YES or NO."

#     results_whole = []
#     results = []
#     technical_abstracts = []
#     non_technical_abstracts = []
#     gold_ips_all = []
#     pred_ips_all = []
#     preds = []
#     gen_only_results = []

#     for i in tqdm(range(len(test_dataset))):
        
#         if debug and i % 100 != 0:
#             continue

#         save_path = None
#         if save_dir is not None:
#             save_path = os.path.join(save_dir, f"{i}.json")
#             preds_save_path = os.path.join(save_dir, f"{i}_preds.json")
#             if os.path.exists(save_path):
#                 scores = json.load(open(save_path))
#                 concise_scores = {
#                     'precision': scores['precision'],
#                     'recall': scores['recall'],
#                     'fscore': scores['fscore']
#                 }
#                 results.append(concise_scores)
#                 pred_dict = json.load(open(preds_save_path))
#                 results.append(pred_dict['results_whole'])
#                 preds.append(pred_dict['preds'])
#                 technical_abstracts.append(pred_dict['technical_abstracts'])
#                 non_technical_abstracts.append(pred_dict['non_technical_abstracts'])
#                 gold_ips_all.append(pred_dict['gold_ips_all'])
#                 pred_ips_all.append(pred_dict['pred_ips_all'])

#                 continue
#         inputs = tokenizer(test_dataset[i]['text'], return_tensors = "pt").to("cuda")

#         outputs = model.generate(**inputs, max_new_tokens = 500, use_cache = True)
#         # import pdb; pdb.set_trace()
#         if 'instruct' in model.config._name_or_path.lower():
#             pred = tokenizer.decode(outputs[0]).replace(test_dataset[i]['text'], '').strip()
#         else:
#             pred = tokenizer.batch_decode(outputs)[0].split('### Response:')[1].strip()
            
#         if gen_only:
#             # preds.append(pred)
#             gen_only_results.append({
#                 'pred': pred,
#                 'technical_abstract': test_dataset[i]['technical_abstract'],
#                 'non_technical_abstract': test_dataset[i]['non_technical_abstract'],
#                 'investigation_proposals': test_dataset[i]['investigation_proposals']
#             })
#             continue

#         pred_ips = text2json(pred)
#         gold_ips = test_dataset[i]['investigation_proposals']
        
#         try:
#             scores = compare_claims(pred_ips, gold_ips, return_all=True, system_prompt=system_prompt, threshold=0.9)
#         except:
#             import pdb; pdb.set_trace()
#             scores = compare_claims(pred_ips, gold_ips, return_all=True, system_prompt=system_prompt, threshold=0.9)
#         concise_scores = {
#             'precision': scores['precision'],
#             'recall': scores['recall'],
#             'fscore': scores['fscore']
#         }

#         results_whole.append(scores)
#         results.append(concise_scores)
#         preds.append(pred)
#         technical_abstracts.append(test_dataset[i]['technical_abstract'])
#         non_technical_abstracts.append(test_dataset[i]['non_technical_abstract'])
#         gold_ips_all.append(gold_ips)
#         pred_ips_all.append(pred_ips)

#         if save_path is not None:
#             os.makedirs(save_dir, exist_ok=True)
#             with open(save_path, "w") as f:
#                 json.dump(concise_scores, f, indent=4)

#             pred_dict = {
#                 "results_whole": scores, 
#                 "preds": pred, "technical_abstracts": test_dataset[i]['technical_abstract'], 
#                 "non_technical_abstracts": test_dataset[i]['non_technical_abstract'], 
#                 "gold_ips_all": gold_ips, "pred_ips_all": pred_ips}
#             with open(os.path.join(save_dir, f"{i}_preds.json"), "w") as f:
#                 json.dump(pred_dict, f, indent=4)

#     if gen_only:
#         return gen_only_results
#     else:
#         return results, results_whole, preds, technical_abstracts, non_technical_abstracts, gold_ips_all, pred_ips_all


def evaluate_model_ips(model, tokenizer, test_dataset, batch_size=8, debug=False, save_dir=None, gen_only=False, force_output=False):
    """
    Evaluate model investigation proposals using batch processing.
    
    Args:
        model: The language model.
        tokenizer: The tokenizer associated with the model.
        test_dataset: A list (or similar) of examples. Each example should have at least:
            - 'text'
            - 'technical_abstract'
            - 'non_technical_abstract'
            - 'investigation_proposals'
        batch_size (int): Number of examples to process per batch.
        debug (bool): If True, only process examples with indices that are multiples of 100.
        save_dir (str): Directory in which to save per-example or per-batch outputs.
        gen_only (bool): If True, only generate predictions without evaluation.
    
    Returns:
        If gen_only is True: a list of dictionaries with generated predictions and associated fields.
        Otherwise: a tuple of lists containing evaluation results and related fields.
    """
    import os
    import json
    import torch
    from torch.utils.data import Dataset, DataLoader, Subset
    from tqdm import tqdm

    # Wrap the dataset to include indices.
    class IndexedDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return idx, self.dataset[idx]

    # Collate function: separate indices and samples.
    def collate_fn(batch):
        indices, samples = zip(*batch)
        return list(indices), list(samples)

    # Optionally filter if in debug mode.
    indexed_dataset = IndexedDataset(test_dataset)
    if debug:
        debug_indices = [i for i in range(len(test_dataset)) if i % 20 == 0]
        indexed_dataset = Subset(indexed_dataset, debug_indices)

    dataloader = DataLoader(indexed_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    system_prompt = (
        "Check two investigation proposals c1 and c2, if c1 is supported by c2. "
        "If c2 includes all the investigations proposed by c1, but also includes additional proposals, then it should still be supported (YES). "
        "If not all proposed investigations by c1 is included in c2, or if c2 contains investigation actions that conflict with investigation actions in c1, then it should be unsupported (NO). "
        "Answer only as a YES or NO."
    )

    results_whole = []
    results = []
    technical_abstracts = []
    non_technical_abstracts = []
    gold_ips_all = []
    pred_ips_all = []
    preds = []
    gen_only_results = []

    for batch_indices, batch_samples in tqdm(dataloader):
        # If saving gen_only outputs, try to use a batch-level file.
        batch_preds_save_path = None
        if save_dir is not None and gen_only:
            batch_preds_save_path = os.path.join(save_dir, f"batch_{batch_indices[0]}-{batch_indices[-1]}_preds_gen_only.json")
            # if os.path.exists(batch_preds_save_path):
            #     # Skip this batch if already processed.
            #     continue

        indices_to_generate = []
        samples_to_generate = []
        # Check each sample: if a saved file exists (for non-gen_only mode), load it.
        for i, sample in zip(batch_indices, batch_samples):
            if save_dir is not None and not gen_only:
                save_path = os.path.join(save_dir, f"{i}.json")
                preds_save_path = os.path.join(save_dir, f"{i}_preds.json")
                if os.path.exists(save_path) and os.path.exists(preds_save_path):
                    scores = json.load(open(save_path))
                    concise_scores = {
                        'precision': scores['precision'],
                        'recall': scores['recall'],
                        'fscore': scores['fscore']
                    }
                    results.append(concise_scores)
                    pred_dict = json.load(open(preds_save_path))
                    results_whole.append(pred_dict['results_whole'])
                    preds.append(pred_dict['preds'])
                    technical_abstracts.append(pred_dict['technical_abstracts'])
                    non_technical_abstracts.append(pred_dict['non_technical_abstracts'])
                    gold_ips_all.append(pred_dict['gold_ips_all'])
                    pred_ips_all.append(pred_dict['pred_ips_all'])
                    continue  # Skip generation for this sample.
            indices_to_generate.append(i)
            samples_to_generate.append(sample)

        if samples_to_generate:
            # Gather texts for tokenization.
            # texts = [sample['text'] for sample in samples_to_generate]
            # # Set tokenizer parameters to ensure consistent padding.
            # tokenizer.padding_side = 'left'
            # tokenizer.pad_token = tokenizer.eos_token
            # inputs = tokenizer(
            #     texts,
            #     return_tensors="pt",
            #     padding=True,
            #     truncation=True,
            #     add_special_tokens=False
            # ).to("cuda")
            # # Generate outputs.
            # outputs = model.generate(**inputs, max_new_tokens=500, use_cache=True)
            # batch_predictions = tokenizer.batch_decode(outputs)

            # # Process each generated prediction.
            # for j, i in enumerate(indices_to_generate):
            #     sample = samples_to_generate[j]
            #     # Process the output depending on whether the model is an instruct model.
            #     if 'instruct' in model.config._name_or_path.lower():
            #         pred = batch_predictions[j].replace(sample['text'], '').strip()
            #     else:
            #         try:
            #             pred = batch_predictions[j].split('### Response:')[1].strip()
            #         except IndexError:
            #             pred = batch_predictions[j].strip()

            texts = [sample['text'] for sample in samples_to_generate]
            tokenizer.padding_side = 'left'
            if tokenizer.padding_side != 'left':
                print('tokenizer.padding_side', tokenizer.padding_side)
            # print('tokenizer.pad_token', tokenizer.pad_token)
            tokenizer.pad_token = tokenizer.eos_token
            inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True).to("cuda")
            if force_output:
                inputs = {key: val[:,:-1] for key, val in inputs.items()}

                                # add_special_tokens=False
                                # truncation=True
                                # ).to("cuda")
            
            outputs = model.generate(**inputs, max_new_tokens=500) #, use_cache=True)
            batch_predictions = tokenizer.batch_decode(outputs)
            # import pdb; pdb.set_trace()

            # Process each generated prediction.
            for j, i in enumerate(indices_to_generate):
                pred_raw = batch_predictions[j]
                # Process the prediction based on model type.
                if 'instruct' in model.config._name_or_path.lower():
                    if force_output:
                        pred = pred_raw.replace(samples_to_generate[j]['text'][3:-4], '').strip()
                    else:
                        pred = pred_raw.replace(samples_to_generate[j]['text'], '').strip()
                    # pred = pred_raw.replace(samples_to_generate[j]['text'], '').strip()
                else:
                    try:
                        pred = pred_raw.split('### Response:')[1].strip()
                    except IndexError:
                        pred = pred_raw.strip()

                if gen_only:
                    gen_only_results.append({
                        'pred': pred,
                        'technical_abstract': sample['technical_abstract'],
                        'non_technical_abstract': sample.get('non_technical_abstract', None),
                        'investigation_proposals': sample.get('investigation_proposals', None)
                    })
                else:
                    try:
                        pred_ips = text2json(pred)
                    except Exception:
                        import pdb; pdb.set_trace()
                        pred_ips = text2json(pred)
                    gold_ips = sample['investigation_proposals']
                    try:
                        scores = compare_claims(
                            pred_ips, gold_ips,
                            return_all=True,
                            system_prompt=system_prompt,
                            threshold=0.9
                        )
                    except Exception:
                        import pdb; pdb.set_trace()
                        scores = compare_claims(
                            pred_ips, gold_ips,
                            return_all=True,
                            system_prompt=system_prompt,
                            threshold=0.9
                        )
                    concise_scores = {
                        'precision': scores['precision'],
                        'recall': scores['recall'],
                        'fscore': scores['fscore']
                    }

                    results_whole.append(scores)
                    results.append(concise_scores)
                    preds.append(pred)
                    technical_abstracts.append(sample['technical_abstract'])
                    non_technical_abstracts.append(sample['non_technical_abstract'])
                    gold_ips_all.append(gold_ips)
                    pred_ips_all.append(pred_ips)

                    if save_dir is not None:
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{i}.json")
                        preds_save_path = os.path.join(save_dir, f"{i}_preds.json")
                        with open(save_path, "w") as f:
                            json.dump(concise_scores, f, indent=4)
                        pred_dict = {
                            "results_whole": scores,
                            "preds": pred,
                            "technical_abstracts": sample['technical_abstract'],
                            "non_technical_abstracts": sample['non_technical_abstract'],
                            "gold_ips_all": gold_ips,
                            "pred_ips_all": pred_ips
                        }
                        with open(preds_save_path, "w") as f:
                            json.dump(pred_dict, f, indent=4)

            # For gen_only mode, save the batch results.
            if gen_only and save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                pred_dict = {
                    'batch_predictions': batch_predictions,
                    'samples_to_generate': samples_to_generate
                }
                with open(batch_preds_save_path, "w") as f:
                    json.dump(pred_dict, f, indent=4)

    if gen_only:
        return gen_only_results
    else:
        return (results, results_whole, preds, technical_abstracts,
                non_technical_abstracts, gold_ips_all, pred_ips_all)



def evaluate_model_claimsips(model, tokenizer, test_dataset, debug=False, save_dir=None, gen_only=False, force_output=False):
    
    # system_prompt = "Check two scientific claims c1 and c2, if c1 is supported by c2. If c2 includes all the evidences for c1, but also includes additional content, then it should still be supported (YES). If not all information of c1 is included in c2, or if c2 contains information that conflicts with information in c1, then it should be unsupported (NO). Answer only as a YES or NO."
    system_prompt = "Check two investigation proposals c1 and c2, if c1 is supported by c2. If c2 includes all the investigations proposed by c1, but also includes additional proposals, then it should still be supported (YES). If not all proposed investigations by c1 is included in c2, or if c2 contains investigation actions that conflict with investigation actions in c1, then it should be unsupported (NO). Answer only as a YES or NO."

    results_whole = []
    results = []
    technical_abstracts = []
    non_technical_abstracts = []
    gold_ips_all = []
    pred_ips_all = []
    preds = []
    gen_only_results = []

    for i in tqdm(range(len(test_dataset))):
        
        if debug and i % 10 != 0:
            continue

        save_path = None
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{i}.json")
            preds_save_path = os.path.join(save_dir, f"{i}_preds.json")
            if os.path.exists(save_path):
                scores = json.load(open(save_path))
                concise_scores = {
                    'precision': scores['precision'],
                    'recall': scores['recall'],
                    'fscore': scores['fscore']
                }
                results.append(concise_scores)
                pred_dict = json.load(open(preds_save_path))
                results.append(pred_dict['results_whole'])
                preds.append(pred_dict['preds'])
                technical_abstracts.append(pred_dict['technical_abstracts'])
                non_technical_abstracts.append(pred_dict['non_technical_abstracts'])
                gold_ips_all.append(pred_dict['gold_ips_all'])
                pred_ips_all.append(pred_dict['pred_ips_all'])

                continue
        inputs = tokenizer(test_dataset[i]['text'], return_tensors = "pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens = 1500, use_cache = True)
        # import pdb; pdb.set_trace()
        if 'instruct' in model.config._name_or_path.lower():
            pred = tokenizer.decode(outputs[0]).replace(test_dataset[i]['text'], '').strip()
        else:
            pred = tokenizer.batch_decode(outputs)[0].split('### Response:')[1].strip()
            
        if gen_only:
            # preds.append(pred)
            gen_only_results.append({
                'pred': pred,
                'technical_abstract': test_dataset[i]['technical_abstract'],
                'non_technical_abstract': test_dataset[i]['non_technical_abstract'],
                'verifiable_claims': test_dataset[i]['verifiable_claims'],
                'investigation_proposals': test_dataset[i]['investigation_proposals']
            })
            continue

        pred_ips = text2json(pred)
        gold_ips = test_dataset[i]['investigation_proposals']
        
        try:
            scores = compare_claims(pred_ips, gold_ips, return_all=True, system_prompt=system_prompt, threshold=0.9)
        except:
            import pdb; pdb.set_trace()
            scores = compare_claims(pred_ips, gold_ips, return_all=True, system_prompt=system_prompt, threshold=0.9)
        concise_scores = {
            'precision': scores['precision'],
            'recall': scores['recall'],
            'fscore': scores['fscore']
        }

        results_whole.append(scores)
        results.append(concise_scores)
        preds.append(pred)
        technical_abstracts.append(test_dataset[i]['technical_abstract'])
        non_technical_abstracts.append(test_dataset[i]['non_technical_abstract'])
        gold_ips_all.append(gold_ips)
        pred_ips_all.append(pred_ips)

        if save_path is not None:
            os.makedirs(save_dir, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(concise_scores, f, indent=4)

            pred_dict = {
                "results_whole": scores, 
                "preds": pred, "technical_abstracts": test_dataset[i]['technical_abstract'], 
                "non_technical_abstracts": test_dataset[i]['non_technical_abstract'], 
                "gold_ips_all": gold_ips, "pred_ips_all": pred_ips}
            with open(os.path.join(save_dir, f"{i}_preds.json"), "w") as f:
                json.dump(pred_dict, f, indent=4)

    if gen_only:
        return gen_only_results
    else:
        return results, results_whole, preds, technical_abstracts, non_technical_abstracts, gold_ips_all, pred_ips_all
  

# if some claims are not used, then we will skip them when computing precision and recall.
def compute_precision_from_results(data_results, return_all=False):
    pred_claim_i_all = set()
    gold_claim_i_all = set()
    for pred_claim_i, gold_claim_i in data_results:
        pred_claim_i_all.add(pred_claim_i)
        gold_claim_i_all.add(gold_claim_i)
    pred_claim_i_all = list(pred_claim_i_all)
    gold_claim_i_all = list(gold_claim_i_all)
        
    if len(pred_claim_i_all) == 0 and len(gold_claim_i_all) == 0:
        return 0
    
    supports = np.zeros((len(pred_claim_i_all), len(gold_claim_i_all)))
    for pi in range(len(pred_claim_i_all)):
        for gi in range(len(gold_claim_i_all)):
            pii = pred_claim_i_all[pi]
            gii = gold_claim_i_all[gi]
            supports[pi, gi] = data_results.get(tuple([pii, gii]), False)
    
    precision_scores = supports.max(-1)
    
    if return_all:
        return precision_scores.mean(), supports
    return precision_scores.mean()

def compute_recall_from_results(data_results, return_all=False):
    pred_claim_i_all = set()
    gold_claim_i_all = set()
    for pred_claim_i, gold_claim_i in data_results:
        pred_claim_i_all.add(pred_claim_i)
        gold_claim_i_all.add(gold_claim_i)
    pred_claim_i_all = list(pred_claim_i_all)
    gold_claim_i_all = list(gold_claim_i_all)
        
    if len(pred_claim_i_all) == 0 and len(gold_claim_i_all) == 0:
        return 0
    
    supports = np.zeros((len(pred_claim_i_all), len(gold_claim_i_all)))
    for pi in range(len(pred_claim_i_all)):
        for gi in range(len(gold_claim_i_all)):
            pii = pred_claim_i_all[pi]
            gii = gold_claim_i_all[gi]
            supports[pi, gi] = data_results.get(tuple([pii, gii]), False)
    
    recall_scores = supports.max(0)
    
    if return_all:
        return recall_scores.mean(), supports
    return recall_scores.mean()