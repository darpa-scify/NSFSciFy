from openai import OpenAI
import json
import os
from dotenv import load_dotenv
from llm_utils import cache
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import re

def parse_final_output(text):
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
    def parse_num(s):
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

    # 7) Claim‐list parser
    def parse_claims(header):
        nonlocal idx
        claims = []
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


@cache.memoize()
def classify_claim(claim: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert at classifying scientifc claims. You are given claim text and asked to classify the claim text into one of the following categories:\n\n1. Established Scientific Fact/Principle -- Claims stating well-accepted scientific laws, principles, or widely known facts.\n2. Observed Phenomenon/Property -- Claims describing observed natural phenomena or material properties.\n3. Capability/Application of Technology/Method -- Claims describing the function, potential, or use of a specific technology, method, or material.\n4. Hypothesis/Theoretical Prediction -- Claims proposing a theoretical explanation, prediction, or hypothesis requiring verification.\n5. Experimental Result/Finding/Measurability -- Claims reporting specific results/findings or stating measurability.\n6. Statement of Problem/Knowledge Gap -- Claims highlighting an existing problem, limitation, or gap in current knowledge or technology.\n7. Definition/Classification -- Claims defining a term or classifying something.\n8. Process/Mechanism Description -- Claims describing a physical or chemical process or mechanism.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{claim}",
                    }
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "claim_category",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "Established Scientific Fact/Principle",
                                "Observed Phenomenon/Property",
                                "Capability/Application of Technology/Method",
                                "Hypothesis/Theoretical Prediction",
                                "Experimental Result/Finding/Measurability",
                                "Statement of Problem/Knowledge Gap",
                                "Definition/Classification",
                                "Process/Mechanism Description",
                            ],
                            "description": "The specific claim category that is being identified.",
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
        raise Exception(  # pylint: disable=broad-exception-raised
            f"Error parsing response. Got: {response.choices[0].message.content}"
        ) from e


import pandas as pd

def summary_to_latex_table(df_summary):
    """
    Given df_summary (index=category, columns=['covered','uncovered']),
    returns a LaTeX table with:
      - '% Covered' column,
      - bold on the % Covered cell for the category with highest % Covered,
      - underline on the Category for the category with highest % Uncovered,
      - caption noting these highlights.
    """
    scientific_order = [
        'Established Scientific Fact/Principle',
        'Observed Phenomenon/Property',
        'Capability/Application of Technology/Method',
        'Hypothesis/Theoretical Prediction',
        'Experimental Result/Finding/Measurability',
        'Statement of Problem/Knowledge Gap',
        'Definition/Classification',
        'Process/Mechanism Description'
    ]
    # Determine order and reindex
    others = [cat for cat in df_summary.index if cat not in scientific_order]
    final_order = scientific_order + others
    df = df_summary.reindex(final_order).fillna(0).astype(int)

    # Compute percentages
    total = df['covered'] + df['uncovered']
    pct_cov = (df['covered'] / total.replace(0,1) * 100).round(1)
    pct_uncov = (df['uncovered'] / total.replace(0,1) * 100).round(1)

    # Identify highlight categories
    best_pct_cov_cat = pct_cov.idxmax()
    best_pct_uncov_cat = pct_uncov.idxmax()

    # Build output DataFrame
    df_out = df.reset_index().rename(columns={'index':'Category','covered':'Covered','uncovered':'Uncovered'})
    # Format '% Covered' column
    def fmt_pct(row):
        cat = row['Category']
        pct = f"{pct_cov[cat]:.1f}\\%"
        if cat == best_pct_cov_cat:
            return f"\\textbf{{{pct}}}"
        return pct

    # Apply formatting
    df_out['% Covered'] = df_out.apply(fmt_pct, axis=1)

    # Underline category name for highest uncovered %
    def fmt_cat(cat):
        if cat == best_pct_uncov_cat:
            return f"\\underline{{{cat}}}"
        return cat

    df_out['Category'] = df_out['Category'].apply(fmt_cat)

    # Caption notes
    caption = (
        "Summary of Claim Coverage by Category. "
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
        label='tab:claim_coverage_summary'
    )

