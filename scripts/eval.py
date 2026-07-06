#!/usr/bin/env python
import argparse
import os
import json

from path_utils import (
    add_src_to_path,
    configure_hf_home,
    get_default_root,
    load_repo_env,
    resolve_model_path,
    resolve_root,
    resolve_under_root,
)

load_repo_env()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a single language model on the NSF Awards dataset."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=get_default_root(),
        help="Root directory for dataset and model files."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Meta-Llama-3.1-8B",
        help="Model name or path to evaluate (e.g., 'unsloth/Meta-Llama-3.1-8B' or a local path)."
    )
    # parser.add_argument(
    #     "--prompt_mode",
    #     type=str,
    #     choices=["tech2nontech", "tech2nontech_instruct", "tech2nontech_instruct_user_assistant"],
    #     default="tech2nontech",
    #     help="Prompt mode for dataset processing."
    # )
    parser.add_argument('--prompt_mode', type=str, choices=['tech2nontech', 'tech2nontech_instruct', 'tech2nontech_instruct_user_assistant',
            'technontech2claims', 'technontech2claims_instruct', 'technontech2claims_instruct_user_assistant', 'text2claims_instruct', 'text2claims_instruct_user_assistant',
            'technontech2ip', 'technontech2ip_instruct', 'technontech2ip_instruct_user_assistant', 'text2ip_instruct', 'text2ip_instruct_user_assistant',
            'technontech2claimsip', 'technontech2claimsip_instruct', 'technontech2claimsip_instruct_user_assistant'], default='tech2nontech')
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="Directory where evaluation results will be saved."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (e.g., for more verbose logging or a smaller data subset)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="tech2nontech",
        choices=["tech2nontech", "technontech2claims", "technontech2ip", "technontech2claimsip"],
        help="Which task we are solving."
    )
    parser.add_argument(
        "--gen_only",
        action="store_true",
        help="Generate predictions only, without evaluation."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mat_sci",
        choices=["mat_sci", "all", "matsci_and_20k", "20k", "matsci"],
        help="Which dataset to evaluate on."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    root_dir = resolve_root(args.root_dir)
    configure_hf_home(root_dir)
    add_src_to_path()

    from data_utils import get_nsf_data_proc, get_nsf_data_all_proc, get_nsf_data_matsci_and_20k_proc, get_nsf_data_20k_proc
    from model_utils import get_model
    from eval_utils import evaluate_model, evaluate_model_claims, evaluate_model_ips, evaluate_model_claimsips

    print("All arguments:", vars(args))
    print(f"Resolved root directory: {root_dir}")

    # Append a debug suffix to the save directory if debug mode is enabled.
    save_dir = resolve_under_root(root_dir, args.save_dir) / (
        args.task + ("_debug" if args.debug else "") + (f"_{args.dataset}" if args.dataset == "all" else "")
    )
    print(f"Saving results to: {save_dir}")
    # save_dir = args.save_dir + ("_debug" if args.debug else "")
    os.makedirs(save_dir, exist_ok=True)

    model_name = resolve_model_path(args.model, root_dir)
    print(f"Loading model from: {model_name}")
    model, tokenizer = get_model(model_name, mode="eval")
    print("Model loaded successfully.")

    print("Processing dataset...")
    # get_nsf_data_proc can return two or three datasets.
    # Here we assume that the test dataset is either the second item (if two are returned)
    # or the third (if three are returned).
    force_output = True if args.dataset == "all" else False
    if args.dataset in ["mat_sci", "matsci"]:
        datasets = get_nsf_data_proc(root_dir, tokenizer, mode=args.prompt_mode, eval_mode=True)
        if len(datasets) == 2:
            _, test_dataset = datasets
        elif len(datasets) >= 3:
            test_dataset = datasets[2]
        else:
            raise ValueError("Unexpected dataset format returned from get_nsf_data_proc.")
        print(f"Loaded {len(test_dataset)} examples from the materials science dataset.")
    elif args.dataset == "matsci_and_20k":
        datasets = get_nsf_data_matsci_and_20k_proc(root_dir, tokenizer, mode=args.prompt_mode, eval_mode=True)
        if len(datasets) == 2:
            _, test_dataset = datasets
        elif len(datasets) >= 3:
            test_dataset = datasets[2]
        else:
            raise ValueError("Unexpected dataset format returned from get_nsf_data_matsci_and_20k_proc.")
        print(f"Loaded {len(test_dataset)} examples from the full dataset.")
    elif args.dataset == "20k":
        datasets = get_nsf_data_20k_proc(root_dir, tokenizer, mode=args.prompt_mode, eval_mode=True, force_output=force_output)
        test_dataset = datasets[2] if len(datasets) >= 3 else datasets[-1]
        print(f"Loaded {len(test_dataset)} examples from the 20k dataset.")
    elif args.dataset == "all":
        test_dataset = get_nsf_data_all_proc(root_dir, tokenizer, mode=args.prompt_mode, eval_mode=True, force_output=force_output)
        print(f"Loaded {len(test_dataset)} examples from the full dataset.")
    print("Dataset processed successfully.")


    # Use the "short" model name (the last part after '/') for naming the result file.
    model_short = args.model.split("/")[-1]
    result_path = os.path.join(save_dir, model_short + ".json")
    preds_path = os.path.join(save_dir, model_short + "_preds.json")
    preds_dir = os.path.join(save_dir, model_short + '_preds')

    gen_only_results = []

    if args.task == "tech2nontech":
        print("Running evaluation...")
        results, preds, technical_abstracts, non_technical_abstracts = evaluate_model(model, tokenizer, test_dataset, debug=args.debug)

        pred_dict = {"preds": preds, "technical_abstracts": technical_abstracts, "non_technical_abstracts": non_technical_abstracts}
    elif args.task == "technontech2claims":
        print("Running evaluation...")
        if args.gen_only:
            gen_only_results = evaluate_model_claims(model, tokenizer, test_dataset, batch_size=args.batch_size, debug=args.debug, 
                                                    save_dir=preds_dir, gen_only=args.gen_only, force_output=force_output)

            os.makedirs(preds_dir, exist_ok=True)
            # save to jsonl
            with open(os.path.join(preds_dir, 'gen_only_results.jsonl'), 'w') as f:
                for item in gen_only_results:
                    f.write(json.dumps(item) + '\n')
            print(f"Saved gen_only_results to: {os.path.join(preds_dir, 'gen_only_results.jsonl')}")

            return
        else:
            results, results_whole, preds, technical_abstracts, non_technical_abstracts, gold_claims_all, pred_claims_all = evaluate_model_claims(model, 
                tokenizer, test_dataset, debug=args.debug, save_dir=preds_dir, gen_only=args.gen_only)
    
        pred_dict = {
            "results_whole": results_whole, 
            "preds": preds, "technical_abstracts": technical_abstracts, 
            "non_technical_abstracts": non_technical_abstracts, "gold_claims_all": gold_claims_all, "pred_claims_all": pred_claims_all}
    elif args.task == "technontech2ip":
        print("Running evaluation...")
        if args.gen_only:
            gen_only_results = evaluate_model_ips(model, tokenizer, test_dataset, batch_size=args.batch_size, debug=args.debug, 
                                                save_dir=preds_dir, gen_only=args.gen_only, force_output=force_output)

            os.makedirs(preds_dir, exist_ok=True)
            # save to jsonl
            with open(os.path.join(preds_dir, 'gen_only_results.jsonl'), 'w') as f:
                for item in gen_only_results:
                    f.write(json.dumps(item) + '\n')
            print(f"Saved gen_only_results to: {os.path.join(preds_dir, 'gen_only_results.jsonl')}")

            return
        else:
            results, results_whole, preds, technical_abstracts, non_technical_abstracts, gold_ips_all, pred_ips_all = evaluate_model_ips(model, 
                tokenizer, test_dataset, debug=args.debug, save_dir=preds_dir, gen_only=args.gen_only)
    
            pred_dict = {
                "results_whole": results_whole, 
                "preds": preds, "technical_abstracts": technical_abstracts, 
                "non_technical_abstracts": non_technical_abstracts, "gold_ips_all": gold_ips_all, "pred_ips_all": pred_ips_all}
    elif args.task == "technontech2claimsip":
        print("Running evaluation...")
        if args.gen_only:
            gen_only_results = evaluate_model_claimsips(model, tokenizer, test_dataset, debug=args.debug, save_dir=preds_dir, gen_only=args.gen_only)

            os.makedirs(preds_dir, exist_ok=True)
            # save to jsonl
            with open(os.path.join(preds_dir, 'gen_only_results.jsonl'), 'w') as f:
                for item in gen_only_results:
                    f.write(json.dumps(item) + '\n')
            print(f"Saved gen_only_results to: {os.path.join(preds_dir, 'gen_only_results.jsonl')}")

            return
        else:
            raise NotImplementedError("technontech2claimsip not implemented yet")
            # results, results_whole, preds, technical_abstracts, non_technical_abstracts, gold_ips_all, pred_ips_all = evaluate_model_ips(model, 
            #     tokenizer, test_dataset, debug=args.debug, save_dir=preds_dir, gen_only=args.gen_only)
    
            # pred_dict = {
            #     "results_whole": results_whole, 
            #     "preds": preds, "technical_abstracts": technical_abstracts, 
            #     "non_technical_abstracts": non_technical_abstracts, "gold_ips_all": gold_ips_all, "pred_ips_all": pred_ips_all}
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    
    # torch.save(results, result_path)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved evaluation results to: {result_path}")

    
    with open(preds_path, "w") as f:
        json.dump(pred_dict, f, indent=4)
    print(f"Saved model predictions to: {preds_path}")

if __name__ == "__main__":
    main()
