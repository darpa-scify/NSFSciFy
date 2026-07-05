"""

\text{Precision} = \frac{1}{|S|} \sum_{c \in S} \max_{g \in G} \text{is\_equivalent}(c, g)


\text{Recall} = \frac{1}{|G|} \sum_{g \in G} \max_{c \in S} \text{is\_equivalent}(g, c)


"""

# THRESHOLD = 0.9
from typing import Tuple
from openai import OpenAI
from dotenv import load_dotenv
import os
import math
import random
import time

from scify_api_keys import OPENAI_API_KEY

load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=OPENAI_API_KEY)

def llm_score(c1, c2, system_prompt=None) -> Tuple[str, float]:
    if system_prompt is None:
        system_prompt = "Check if two scientific claims c1 and c2 are equivalent. Answer only as a YES or NO."
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"c1: {c1}\nc2: {c2}"}],
            },
        ],
        response_format={"type": "text"},
        temperature=0,
        max_completion_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logit_bias={31958: 100, 14695: 100},
        logprobs=True,
    )
    completion = response.choices[0].logprobs.content[0].token.strip().lower()
    logprob = response.choices[0].logprobs.content[0].logprob
    sleep_duration = random.uniform(0.5, 2)
    time.sleep(sleep_duration)
    return completion, math.exp(logprob)


def is_supported_by(c1, c2, system_prompt=None, threshold=0.9) -> Tuple[bool, str, float]:
    completion, logprob = llm_score(c1, c2, system_prompt=system_prompt)
    return completion == "yes" and logprob >= threshold, completion, logprob


def compute_precision(predicted_claims, gold_claims, return_all=False, system_prompt=None, threshold=0.9):
    if not predicted_claims:
        return 0.0

    precision_scores = []
    precision_scores_logprob = []
    precision_scores_all = []
    precision_scores_completion_all = []
    precision_scores_logprob_all = []
    
    for claim in predicted_claims:
        current_scores = []
        current_scores_completion = []
        current_scores_logprob = []
        for gold in gold_claims:
            score, completion, logprob = is_supported_by(claim, gold, system_prompt=system_prompt, threshold=threshold)
            current_scores.append(score)
            current_scores_completion.append(completion)
            current_scores_logprob.append(logprob)
        max_score = max(current_scores) if current_scores else 0.0
        max_score_logprob = max(current_scores_logprob) if current_scores_logprob else 0.0
        precision_scores.append(max_score)
        precision_scores_logprob.append(max_score_logprob)
        precision_scores_all.append(current_scores)
        precision_scores_completion_all.append(current_scores_completion)
        precision_scores_logprob_all.append(current_scores_logprob)

    if return_all:
        return sum(precision_scores) / len(predicted_claims), precision_scores, precision_scores_logprob, precision_scores_all, precision_scores_completion_all, precision_scores_logprob_all
    else:
        return sum(precision_scores) / len(predicted_claims)


def compute_recall(predicted_claims, gold_claims, return_all=False, system_prompt=None, threshold=0.9):
    if not gold_claims:
        return 0.0

    recall_scores = []
    recall_scores_logprob = []
    recall_scores_all = []
    recall_scores_completion_all = []
    recall_scores_logprob_all = []

    for gold in gold_claims:
        current_scores = []
        current_scores_completion = []
        current_scores_logprob = []
        for claim in predicted_claims:
            score, completion, logprob = is_supported_by(gold, claim, system_prompt=system_prompt, threshold=threshold)
            current_scores.append(score)
            current_scores_completion.append(completion)
            current_scores_logprob.append(logprob)
        max_score = max(current_scores) if current_scores else 0.0
        max_score_logprob = max(current_scores_logprob) if current_scores_logprob else 0.0
        recall_scores.append(max_score)
        recall_scores_logprob.append(max_score_logprob)
        recall_scores_all.append(current_scores)
        recall_scores_completion_all.append(current_scores_completion)
        recall_scores_logprob_all.append(current_scores_logprob)

    if return_all:
        return sum(recall_scores) / len(gold_claims), recall_scores, recall_scores_logprob, recall_scores_all, recall_scores_completion_all, recall_scores_logprob_all
    else:
        return sum(recall_scores) / len(gold_claims)

def compare_claims(predicted_claims, gold_claims, return_all=False, system_prompt=None, threshold=0.9) -> dict:
    precision_output = compute_precision(predicted_claims, gold_claims, return_all=return_all, system_prompt=system_prompt, threshold=threshold)
    recall_output = compute_recall(predicted_claims, gold_claims, return_all=return_all, system_prompt=system_prompt, threshold=threshold)

    if return_all:
        precision, precision_scores, precision_scores_logprob, precision_scores_all, precision_scores_completion_all, precision_scores_logprob_all = precision_output
        recall, recall_scores, recall_scores_logprob, recall_scores_all, recall_scores_completion_all, recall_scores_logprob_all = recall_output
        fscore = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
            "precision_scores_logprob": precision_scores_logprob,
            "recall_scores_logprob": recall_scores_logprob,
            "precision_scores_all": precision_scores_all,
            "recall_scores_all": recall_scores_all,
            "precision_scores_completion_all": precision_scores_completion_all,
            "recall_scores_completion_all": recall_scores_completion_all,
            "precision_scores_logprob_all": precision_scores_logprob_all,
            "recall_scores_logprob_all": recall_scores_logprob_all,
        }
    else:
        precision, recall = precision_output, recall_output
        fscore = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "fscore": fscore}