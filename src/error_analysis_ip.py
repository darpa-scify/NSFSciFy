
"""
error_analysis_ip.py
--------------------
Utilities for error analysis of INVESTIGATION_PROPOSALS.

Updates:
  * Classifier categories are now LOADED from a JSON file under the repository,
    rather than hard-coded. The OpenAI JSON schema and the system prompt are constructed dynamically
    from that file.

File schema (ip_categories.json):
    {
      "Category Name A": "Short description of category A",
      "Category Name B": "Short description of category B",
      ...
    }
"""

from __future__ import annotations

import os
import re
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

# OpenAI client
try:
    from openai import OpenAI
    from llm_utils import cache  # user-provided util with @memoize
    _HAS_OPENAI = True
except Exception:
    # Allow import of file without OpenAI installed (for formatting-only tasks)
    _HAS_OPENAI = False
    class DummyCache:
        def memoize(self, *args, **kwargs):
            def deco(f):
                return f
            return deco
    cache = DummyCache()  # type: ignore

BASE_DIR = os.path.dirname(__file__)
DEFAULT_CATEGORY_FILE = os.path.join(BASE_DIR, "../notebooks/error_analysis/tasks/ip_categories.json")

# ----------------------
# Category I/O helpers
# ----------------------
def load_ip_categories(path: Optional[str] = None) -> Dict[str, str]:
    """
    Load categories from a JSON file mapping {category_name: description}.
    """
    path = path or DEFAULT_CATEGORY_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(f"Category file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not obj:
        raise ValueError("ip_categories.json must be a non-empty object mapping names to descriptions")
    # Ensure keys are strings
    return {str(k): str(v) for k, v in obj.items()}

_CATEGORIES = load_ip_categories()
_CATEGORY_NAMES = list(_CATEGORIES.keys())
_CATEGORY_DESCRIPTIONS = _CATEGORIES  # alias

def categories_as_prompt_text() -> str:
    lines = ["You must classify each proposal into exactly one of the following categories:"]
    for i, name in enumerate(_CATEGORY_NAMES, 1):
        desc = _CATEGORY_DESCRIPTIONS.get(name, "").strip()
        if desc:
            lines.append(f"{i}. {name} — {desc}")
        else:
            lines.append(f"{i}. {name}")
    return "\n".join(lines)

# -------------
# Parsing utils
# -------------

def parse_final_output(text: str) -> Dict:
    """
    Find the FINAL OUTPUT section, discard any lines containing backticks,
    and parse Precision, Recall (fractions/percentages/floats), two claim lists,
    and the summary counts.
    """
    lines_all = text.splitlines()

    # 1) Locate the FINAL OUTPUT marker
    start_idx = None
    for i, ln in enumerate(lines_all):
        if ln.strip() == "FINAL OUTPUT":
            start_idx = i
            break
    if start_idx is None:
        for i, ln in enumerate(lines_all):
            if "final output" in ln.lower():
                start_idx = i
                break
    if start_idx is None:
        raise ValueError("Could not find a line with 'FINAL OUTPUT'")

    # 2) Take everything after that line
    after_lines = lines_all[start_idx+1:]

    # 3) Remove any line that contains backticks, and trim whitespace
    lines = [ln.strip() for ln in after_lines if "```" not in ln and ln.strip()]

    data = {}
    idx = 0

    # 4) Enhanced number parser
    def parse_num(s: str) -> float:
        s = s.strip()
        # fraction "n/d"
        if "/" in s and re.match(r"^\s*\d+\s*/\s*\d+\s*$", s):
            num, den = s.split("/", 1)
            return float(num) / float(den)
        # percentage "xx%"
        if s.endswith("%"):
            return float(s.rstrip("%")) / 100.0
        return float(s)

    # 5) Precision
    if idx < len(lines) and lines[idx].lower().startswith("precision"):
        _, val = lines[idx].split(":", 1)
        data["Precision"] = parse_num(val)
        idx += 1

    # 6) Recall
    if idx < len(lines) and lines[idx].lower().startswith("recall"):
        _, val = lines[idx].split(":", 1)
        data["Recall"] = parse_num(val)
        idx += 1

    # 7) Claim-list parser
    def parse_claims(header: str) -> List[Dict[str, str]]:
        nonlocal idx
        claims: List[Dict[str, str]] = []
        if idx < len(lines) and lines[idx].startswith(header):
            idx += 1
            while idx < len(lines) and re.match(r"\d+\.", lines[idx]):
                _, rest = lines[idx].split(".", 1)
                if "category:" in rest:
                    claim_part, cat_part = rest.split("category:", 1)
                    claim_text = claim_part.replace("claim:", "").strip().rstrip(",")
                    category   = cat_part.strip()
                    claims.append({"claim": claim_text, "category": category})
                idx += 1
        return claims

    data["Extracted claims"] = parse_claims("Extracted claims")
    data["Extra claims"]     = parse_claims("Extra claims")

    # 8) Summary
    summary = {}
    if idx < len(lines) and lines[idx].lower().startswith("summary"):
        idx += 1
        while idx < len(lines):
            m = re.match(r"(.*):\s*covered:\s*(\d+),\s*uncovered:\s*(\d+)", lines[idx])
            if m:
                cat = m.group(1).strip()
                summary[cat] = {"covered": int(m.group(2)), "uncovered": int(m.group(3))}
            idx += 1
    if summary:
        data["Summary"] = summary

    return data

# ----------------------
# Claim categorization
# ----------------------

if _HAS_OPENAI:
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    _client = None  # type: ignore

def _build_classifier_messages(claim: str) -> Tuple[List[dict], dict]:
    """
    Construct messages and response_format JSON schema based on loaded categories.
    """
    system_text = (
        "You are an expert at classifying investigation proposals into functional categories.\n"
        + categories_as_prompt_text()
        + "\nReturn JSON strictly following the provided schema."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "text", "text": claim}]},
    ]
    schema = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": _CATEGORY_NAMES,
                "description": "One of the category names from ip_categories.json",
            }
        },
        "required": ["category"],
        "additionalProperties": False,
    }
    response_format = {"type": "json_schema", "json_schema": {"name": "ip_category", "strict": True, "schema": schema}}
    return messages, response_format

@cache.memoize()
def classify_claim(claim: str) -> str:
    """
    Classify an investigation proposal into one of the categories loaded from ip_categories.json.
    Always uses gpt-4o-mini; no heuristics or fallback.
    """
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an expert at classifying investigation proposals. "
                        "You are given a proposal text and must classify it into exactly one of the following categories:\n\n"
                        + "\n".join(
                            f"{i+1}. {name} -- {desc}"
                            for i, (name, desc) in enumerate(_CATEGORIES.items())
                        )
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": claim}],
        },
    ]

    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ip_category",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": _CATEGORY_NAMES,
                            "description": "One of the category names from ip_categories.json",
                        }
                    },
                    "required": ["category"],
                    "additionalProperties": False,
                },
            },
        },
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        store=False,
    )
    try:
        return json.loads(response.choices[0].message.content)["category"]
    except Exception as e:
        print(response.choices[0].message.content)
        raise Exception(
            f"Error parsing response. Got: {response.choices[0].message.content}"
        ) from e


# ------------------------------
# Summary aggregation utilities
# ------------------------------

def build_summary(df_claims: pd.DataFrame, gold: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """
    Build a per-category summary DataFrame with columns ['covered','uncovered'].
    """
    if "category" not in df_claims.columns:
        raise ValueError("df_claims must have a 'category' column")

    pred_counts = df_claims["category"].value_counts().to_dict()
    categories = set(pred_counts.keys())
    if gold is not None:
        categories |= set(gold.keys())

    covered = {}
    uncovered = {}

    if gold is None:
        for cat in categories:
            covered[cat] = int(pred_counts.get(cat, 0))
            uncovered[cat] = 0
    else:
        gold_sets: Dict[str, set] = {cat: set(v) for cat, v in gold.items()}
        by_cat = df_claims.groupby("category")["proposal"].apply(list).to_dict()
        for cat in categories:
            pred_set = set(by_cat.get(cat, []))
            gold_set = gold_sets.get(cat, set())
            inter = pred_set & gold_set
            covered[cat] = int(len(inter))
            uncovered[cat] = int(len(gold_set - pred_set))

    df = pd.DataFrame({"covered": pd.Series(covered), "uncovered": pd.Series(uncovered)})
    df = df.fillna(0).astype(int).sort_index()
    return df

def summary_to_latex_table(df_summary: pd.DataFrame) -> str:
    """
    Render LaTeX table highlighting best % covered and worst % uncovered.
    Order respects the category file order first, then others.
    """
    preferred = _CATEGORY_NAMES
    others = [cat for cat in df_summary.index if cat not in preferred]
    final_order = preferred + others
    df = df_summary.reindex(final_order).fillna(0).astype(int)

    total = df['covered'] + df['uncovered']
    pct_cov = (df['covered'] / total.replace(0,1) * 100).round(1)
    pct_uncov = (df['uncovered'] / total.replace(0,1) * 100).round(1)

    best_pct_cov_cat = pct_cov.idxmax()
    best_pct_uncov_cat = pct_uncov.idxmax()

    df_out = df.reset_index().rename(columns={'index':'Category','covered':'Covered','uncovered':'Uncovered'})
    def fmt_pct(row):
        cat = row['Category']
        pct = f"{pct_cov[cat]:.1f}\\%"
        if cat == best_pct_cov_cat:
            return f"\\textbf{{{pct}}}"
        return pct
    df_out['% Covered'] = df_out.apply(fmt_pct, axis=1)

    def fmt_cat(cat):
        if cat == best_pct_uncov_cat:
            return f"\\underline{{{cat}}}"
        return cat
    df_out['Category'] = df_out['Category'].apply(fmt_cat)

    caption = (
        "Summary of Investigation-Proposal Coverage by Category. "
        f"Bold = highest percent covered ({best_pct_cov_cat}); "
        f"Underline = highest percent uncovered ({best_pct_uncov_cat})."
    )
    return df_out.to_latex(
        index=False,
        columns=['Category', 'Covered', 'Uncovered', '% Covered'],
        header=['Category', 'Covered', 'Uncovered', '% Covered'],
        escape=False,
        column_format='lrrr',
        caption=caption,
        label='tab:ip_coverage_summary'
    )

# ------------------------------
# Convenience: end-to-end demo
# ------------------------------

def categorize_investigation_proposals(proposals: List[str]) -> pd.DataFrame:
    """
    Takes a list of proposal strings, classifies each, and returns a DataFrame
    with columns ['proposal', 'category'].
    """
    rows = []
    for p in proposals:
        cat = classify_claim(p)
        rows.append({"proposal": p, "category": cat})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Minimal sanity check without requiring OpenAI:
    sample = [
        "Develop a scalable imaging pipeline for nanoscale materials.",
        "We propose a user study to assess usability of the new visualization.",
        "Benchmark three retrieval algorithms on the XYZ dataset.",
    ]
    df_claims = categorize_investigation_proposals(sample)
    df_sum = build_summary(df_claims)  # no gold provided
    print(df_claims)
    print()
    print(summary_to_latex_table(df_sum))
