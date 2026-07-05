import re
import json
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)
import nltk
nltk.download('punkt_tab')

def remove_invalid_unicode(candidate):
    """
    Removes any \\u escape sequences from candidate that are not exactly 4 valid hex digits.
    Valid escapes are kept.
    """
    def repl(match):
        seq = match.group(0)  # e.g. "\u201" or "\u20G9"
        hex_part = seq[2:]
        if len(hex_part) == 4 and all(c in "0123456789abcdefABCDEF" for c in hex_part):
            # Valid sequence; keep it.
            return seq
        else:
            # Invalid sequence; remove it.
            return ""
    # This pattern matches any \u followed by 1 to 4 characters.
    return re.sub(r'\\u.{1,4}', repl, candidate)

def extract_array(text):
    """
    Extracts the first array from the text.
    It finds the first '[' and searches for a matching ']' using a bracket counter.
    If no array is found, or if JSON parsing fails,
    it falls back to extracting complete quoted elements.
    If that still fails, it uses sentence tokenization as a last resort.
    Also, it fixes invalid backslashes and removes invalid \\u escapes.
    """
    start = text.find('[')
    if start == -1:
        # No array found; fallback to sentence tokenization.
        return sent_tokenize(text)
    
    bracket_count = 0
    end_index = None
    for i in range(start, len(text)):
        if text[i] == '[':
            bracket_count += 1
        elif text[i] == ']':
            bracket_count -= 1
            if bracket_count == 0:
                end_index = i + 1
                break

    if end_index is None:
        # Array seems truncated; candidate is everything from first '[' onward.
        candidate = text[start:]
    else:
        candidate = text[start:end_index]

    # Normalize newlines.
    candidate = candidate.replace('\n', ' ')
    
    # Fix invalid backslashes (if not followed by valid escape characters).
    candidate = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate)
    
    # Remove invalid \u escapes.
    candidate = remove_invalid_unicode(candidate)
    
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        print("Standard JSON decode error:", e)
        # Fallback: extract only complete quoted strings.
        pattern = r'"((?:\\.|[^"\\])*)"'
        matches = re.findall(pattern, candidate)
        valid_json_str = '[' + ', '.join(f'"{m}"' for m in matches) + ']'
        try:
            return json.loads(valid_json_str)
        except json.JSONDecodeError:
            # As a last resort, return tokenized sentences.
            return sent_tokenize(text)

# # Example usage:
# sample_text = 'random text before the array ["foo", "bar", "baz"'
# extracted = extract_array(sample_text)
# print(extracted)

# extract_array('adsfsd dsdf . sdfsd .')


def get_openai_api_call(custom_id, c1, c2, system_prompt, model_name='gpt-4o-mini'):
    single_message = {
        "custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", 
        "body": {
            "model": model_name,
            "messages": [
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
            "response_format": {"type": "text"},
            "temperature": 0,
            "max_completion_tokens": 1,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "logit_bias": {31958: 100, 14695: 100},
            "logprobs": True,
        }
    }
    return single_message