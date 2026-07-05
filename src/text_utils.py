import re
import json

def extract_json_array(text):
    """
    Extracts the first balanced JSON array (including nested arrays)
    by locating the matching closing bracket for the first '['.
    """
    start = text.find('[')
    if start == -1:
        return None
    bracket_count = 0
    for i in range(start, len(text)):
        if text[i] == '[':
            bracket_count += 1
        elif text[i] == ']':
            bracket_count -= 1
            if bracket_count == 0:
                return text[start:i+1]
    return None

def parse_array_of_strings(json_str):
    """
    Custom parser for JSON arrays that are meant to contain only strings.
    This function removes the outer brackets, then iterates over the content.
    While inside a string (after an opening quote), if an unescaped quote is
    encountered and the following non-space character is not a comma or the end,
    it is treated as an interior quote and is escaped.
    """
    s = json_str.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]  # remove outer brackets
    result = []
    i = 0
    while i < len(s):
        # Skip any whitespace.
        while i < len(s) and s[i].isspace():
            i += 1
        if i >= len(s):
            break
        # Expect an opening quote.
        if s[i] != '"':
            i += 1
            continue
        i += 1  # Skip the opening quote.
        element_chars = []
        while i < len(s):
            if s[i] == '\\':
                # Copy escape sequences as-is.
                element_chars.append(s[i])
                if i+1 < len(s):
                    element_chars.append(s[i+1])
                    i += 2
                else:
                    i += 1
            elif s[i] == '"':
                # Check if this quote is the closing quote.
                j = i + 1
                while j < len(s) and s[j].isspace():
                    j += 1
                # If we see a comma or nothing afterward, treat it as a closing quote.
                if j >= len(s) or s[j] == ',':
                    i += 1  # consume the closing quote
                    break
                else:
                    # Otherwise, assume it is an unescaped interior quote.
                    element_chars.append('\\"')
                    i += 1
            else:
                element_chars.append(s[i])
                i += 1
        result.append("".join(element_chars))
        # Skip any whitespace and commas.
        while i < len(s) and (s[i].isspace() or s[i] == ','):
            i += 1
    return result

def text2json(text):
    """
    Extracts JSON arrays from the input text.
    The function first searches for markdown code blocks marked with ```json ... ``` inside <s> ... </s>.
    If found, each block is preprocessed (newlines normalized and a regex fixes missing commas between adjacent string literals)
    and then decoded with json.loads(). If json.loads() fails (for example due to unescaped quotes),
    the function falls back to the custom parser.
    If no markdown block is found, a fallback extraction is attempted.
    """
    # Look for markdown code blocks labeled as json.
    matches = re.findall(r'```json\s*(\[[\s\S]*?\])\s*```', text, re.DOTALL)
    combined_claims = []
    if matches:
        for json_str in matches:
            # Normalize newlines (unescaped newlines replaced with a space).
            json_str = re.sub(r'(?<!\\)\n', ' ', json_str)
            # Optionally insert a comma between adjacent string literals if missing.
            json_str = re.sub(r'(?<!\\)"\s+(?=")', '", ', json_str)
            try:
                arr = json.loads(json_str)
            except json.JSONDecodeError as e:
                print("Standard JSON decode error:", e)
                # Fall back to our custom parser.
                arr = parse_array_of_strings(json_str)
                # import pdb; pdb.set_trace()
            if isinstance(arr, list):
                combined_claims.extend(arr)
            else:
                combined_claims.append(arr)
        return combined_claims
    else:
        # Fallback: try to extract a balanced JSON array anywhere in the text.
        json_str = extract_json_array(text)
        if json_str is None:
            return None
        json_str = re.sub(r'(?<!\\)\n', ' ', json_str)
        json_str = re.sub(r'(?<!\\)"\s+(?=")', '", ', json_str)
        try:
            data_obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            print("Standard JSON decode error:", e)
            data_obj = parse_array_of_strings(json_str)
            # import pdb; pdb.set_trace()
        return data_obj