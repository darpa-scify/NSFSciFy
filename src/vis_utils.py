import numpy as np
import random
import matplotlib.pyplot as plt
import os  # Needed for saving the figure
from collections import defaultdict

def bootstrap_std(values, n_bootstrap=1000, seed=42):
    """
    Compute bootstrap-based standard deviation of the mean.
    values: 1D list or array of metric values.
    n_bootstrap: number of bootstrap samples.
    seed: random seed for reproducibility.
    """
    random.seed(seed)
    values = np.array(values)
    means = []
    n = len(values)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
        
    return np.std(means)

# def aggregate_and_analyze(results):
#     """
#     Takes a list of result dicts and returns a dictionary of
#     {metric_name: (mean, std)} pairs. 
#     """
#     # Prepare containers for each metric
#     metrics = {
#         "bertscore_precision": [],
#         "bertscore_recall": [],
#         "bertscore_f1": [],
#         "rouge1": [],
#         "rouge2": [],
#         "rougeL": [],
#         "rougeLsum": [],
#         "bleu": []
#     }
    
#     # Aggregate all values
#     for res in results:
#         # BERTScore can come in lists; extend or append accordingly
#         metrics["bertscore_precision"].extend(res["bertscore"]["precision"])
#         metrics["bertscore_recall"].extend(res["bertscore"]["recall"])
#         metrics["bertscore_f1"].extend(res["bertscore"]["f1"])
        
#         # ROUGE are single floats
#         metrics["rouge1"].append(res["rouge"]["rouge1"])
#         metrics["rouge2"].append(res["rouge"]["rouge2"])
#         metrics["rougeL"].append(res["rouge"]["rougeL"])
#         metrics["rougeLsum"].append(res["rouge"]["rougeLsum"])
        
#         # BLEU is also a single float
#         metrics["bleu"].append(res["bleu"]["bleu"])
    
#     # Now compute mean and bootstrap std for each metric
#     analysis = {}
#     for metric_name, values in metrics.items():
#         mean_val = np.mean(values)
#         std_val = bootstrap_std(values, n_bootstrap=1000, seed=42)
#         analysis[metric_name] = (mean_val, std_val)
    
#     return analysis

def aggregate_and_analyze(results, metrics_keep=None):
    """
    Takes a list of result dicts and returns a dictionary of
    {metric_name: (mean, std)} pairs. 
    """
    # Prepare containers for each metric
    metrics = defaultdict(list)
    
    for res in results:
        for key, val in res.items():
            if metrics_keep:
                if key not in metrics_keep:
                    continue
            metrics[key].append(val)
    
    # Now compute mean and bootstrap std for each metric
    analysis = {}
    for metric_name, values in metrics.items():
        mean_val = np.mean(values)
        std_val = bootstrap_std(values, n_bootstrap=1000, seed=42)
        analysis[metric_name] = (mean_val, std_val)
    
    return analysis

def analyze(results, metrics_keep=None):
    """
    Takes a result dict and returns a dictionary of
    {metric_name: (mean, std)} pairs. 
    """
    
    # Now compute mean and bootstrap std for each metric
    analysis = {}
    for metric_name, values in results.items():
        mean_val = np.mean(values)
        std_val = bootstrap_std(values, n_bootstrap=1000, seed=42)
        analysis[metric_name] = (mean_val, std_val)
    
    return analysis


def plot_grouped_bar(results):
    """
    Given the results dictionary {model_name: {metric_name: (mean, std)}},
    create a grouped bar plot with error bars for each metric.
    """
    models = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    means = {metric: [results[model][metric][0] for model in models] for metric in metrics}
    stds = {metric: [results[model][metric][1] for model in models] for metric in metrics}

    x = np.arange(len(metrics))  # Label locations
    width = 0.35  # Width of bars

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        ax.bar(x + i * width, [means[m][i] for m in metrics], width, 
               yerr=[stds[m][i] for m in metrics], capsize=5, label=model, alpha=0.7)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Mean Scores with Bootstrap Std for Different Models')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.show()
    

def plot_grouped_bar_adjustable_width(results, save_path=None, name_mapping=None, figsize=(5.5, 5),
        rotation=45, ha='right', title='Mean Scores with 95% CI for Different Models'):
    """
    Given the results dictionary {model_name: {metric_name: (mean, std)}},
    create a grouped bar plot with error bars representing the 95% confidence intervals,
    adjusting the bar width dynamically based on the number of models.
    
    Note: Here we assume the provided 'std' is an estimate of the standard error,
    so the 95% CI is computed as 1.96 * std.
    """
    models = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    means = {metric: [results[model][metric][0] for model in models] for metric in metrics}
    stds = {metric: [results[model][metric][1] for model in models] for metric in metrics}

    x = np.arange(len(metrics))  # Label locations
    width = max(0.1, 0.8 / len(models))  # Dynamically adjust width

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(models):
        # Compute the 95% confidence interval margin as 1.96 * std (assuming std is the standard error)
        ci = [1.96 * stds[m][i] for m in metrics]
        ax.bar(x + i * width - (len(models) - 1) * width / 2,  # Center the bars
               [means[m][i] for m in metrics],
               width, 
               yerr=ci, capsize=5, label=model[:20] if name_mapping is None else name_mapping[model], alpha=0.7)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=rotation, ha=ha)
    ax.legend()
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    
    plt.show()

def precision_plot(precision_scores_logprob_all, precision_scores_completion_all, pred_claims, gold_claims):
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Display the log probability matrix as a heatmap.
    im = ax.imshow(precision_scores_logprob_all, cmap='viridis', vmin=0, vmax=1)

    # Set the tick positions and labels.
    ax.set_xticks(np.arange(len(gold_claims)))
    ax.set_yticks(np.arange(len(pred_claims)))
    ax.set_xticklabels(gold_claims, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(pred_claims, fontsize=12)

    # Loop over each cell to add the corresponding text from precision_scores_completion_all.
    for i in range(precision_scores_logprob_all.shape[0]):
        for j in range(precision_scores_logprob_all.shape[1]):
            text = precision_scores_completion_all[i, j]
            # Set color: green if 'yes', red if 'no'
            color = 'green' if text == 'yes' else 'red'
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    # Add a colorbar for reference and title.
    fig.colorbar(im, ax=ax)
    ax.set_title("Precision Scores with Completion Annotations", fontsize=12)

    plt.tight_layout()
    plt.show()
    
def recall_plot(recall_scores_logprob_all, recall_scores_completion_all, pred_claims, gold_claims):
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Display the log probability matrix as a heatmap.
    im = ax.imshow(recall_scores_logprob_all, cmap='viridis', vmin=0, vmax=1)

    # Set the tick positions and labels.
    ax.set_xticks(np.arange(len(pred_claims)))
    ax.set_yticks(np.arange(len(gold_claims)))
    ax.set_xticklabels(pred_claims, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(gold_claims, fontsize=12)

    # Loop over each cell to add the corresponding text from precision_scores_completion_all.
    for i in range(recall_scores_logprob_all.shape[0]):
        for j in range(recall_scores_logprob_all.shape[1]):
            text = recall_scores_completion_all[i, j]
            # Set color: green if 'yes', red if 'no'
            color = 'green' if text == 'yes' else 'red'
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    # Add a colorbar for reference and title.
    fig.colorbar(im, ax=ax)
    ax.set_title("Recall Scores with Completion Annotations", fontsize=12)

    plt.tight_layout()
    plt.show()