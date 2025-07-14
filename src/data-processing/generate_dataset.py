import ollama
import json
import logging
import re
import time
from json import JSONDecodeError

logging.basicConfig(level=logging.INFO)

# Define the number of samples to generate.
NUM_SAMPLES = 2000
OUTPUT_FILE = 'faulty_vs_fixed_dataset.jsonl'
MAX_RETRIES = 5  # Define a maximum number of retries

# Prompt template for generating faulty and fixed code samples.

PROMPT_TEMPLATE = ("""

Generate a pair of Python code snippets in strict JSON format. The output must be a single JSON object with exactly two fields: 'faulty_code' and 'fixed_code'. Do not include any additional text, explanations, or markdown code blocks.

1. 'faulty_code' must contain a short, medium, or long Python code snippet with a realistic programming bug or defect (such as logic errors, off-by-one errors, incorrect variable names, misuse of variables, etc.).
2. 'fixed_code' must contain the corrected version of the code snippet.
3. The code snippets must be realistic and relevant to common programming tasks.
4. The code snippets must be syntactically correct and executable in Python.
5. The code snippets should be of varying lengths: short (1-3 lines), medium (4-10 lines), and long (11+ lines).
6. Provide a diverse set of examples that cover different types of bugs and fixes.
7. The output must be a single JSON object, with no extra text or formatting.

Example output format:
{
  "faulty_code": "print('Hello, world!')",
  "fixed_code": "print('Hello, world!')\n# This example is very simple; your code should be more realistic."
}

"STRICT FORMATTING RULES:\n"
"1. Wrap the JSON in ```json code blocks\n"
"2. Escape all newlines with \\n\n"
"3. Never use triple backticks outside the JSON wrapper\n"
"4. Ensure both code snippets are non-empty\n"
                   """
)


def parse_model_response(content: str) -> dict:
    """
    Robust parser that handles:
    - JSON wrapped in markdown code blocks
    - Malformed JSON with common errors
    - Multiple JSON candidates in response
    """
    strategies = [
        # Strategy 1: Direct JSON parse
        lambda: json.loads(content),
        
        # Strategy 2: Extract JSON from ```
        lambda: extract_and_parse(r'```(?:json)?\n(.*?)(?=```)', content),
        
        # Strategy 3: Extract JSON from generic ``` blocks
        lambda: extract_and_parse(r'```\n(.*?)(?=```)', content),
        
        # Strategy 4: Find deepest JSON-like structure
        lambda: find_json_in_text(content),
        
        # Strategy 5: Lenient parse with error recovery
        lambda: json.loads(sanitize_json(content))
    ]

    for strategy in strategies:
        try:
            result = strategy()
            if validate_result(result):
                return result
        except (JSONDecodeError, AttributeError, KeyError):
            continue

    logging.error("All parsing strategies failed")
    return None

def extract_and_parse(pattern: str, content: str) -> dict:
    """Extract JSON using regex pattern and parse"""
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    raise JSONDecodeError("No match", "", 0)

def find_json_in_text(content: str) -> dict:
    """Find JSON-like structures using bracket counting"""
    candidates = []
    stack = []
    start_index = -1
    
    for i, char in enumerate(content):
        if char == '{':
            if not stack:
                start_index = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    candidates.append(content[start_index:i+1])
    
    # Try candidates from longest to shortest
    for candidate in sorted(candidates, key=len, reverse=True):
        try:
            return json.loads(candidate)
        except JSONDecodeError:
            continue
    raise JSONDecodeError("No valid JSON found", "", 0)

def sanitize_json(text: str) -> str:
    """Fix common JSON formatting issues"""
    # Remove trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Remove non-JSON content
    text = re.sub(r'^[^{]*', '', text, flags=re.DOTALL)
    text = re.sub(r'[^}]*$', '', text, flags=re.DOTALL)
    return text.strip()

def validate_result(result: dict) -> bool:
    """Validate the parsed JSON structure"""
    return isinstance(result, dict) and \
           'faulty_code' in result and \
           'fixed_code' in result and \
           len(result['faulty_code']) > 0 and \
           len(result['fixed_code']) > 0



def generate_code_samples():

    response = ollama.chat(
    model="deepseek-r1:7b",
    messages=[{"role": "user", "content": PROMPT_TEMPLATE}],
    options={
        'temperature': 0.3,
        'format': 'json',  # Explicitly request JSON format
    }
)   
    # Extract the content string from the response
    content = response['message']['content'] if 'message' in response and 'content' in response['message'] else ""
    return parse_model_response(content)

def main():
    samples = []
    for i in range(NUM_SAMPLES):
        sample = None
        retries = 0
        while sample is None and retries < MAX_RETRIES:
            sample = generate_code_samples()
            retries += 1
            if sample is None:
                logging.warning("Retrying due to parsing error...")
            time.sleep(1)
        if sample is None:
            logging.error(f"Failed to generate sample after {MAX_RETRIES} retries.")
            continue
        samples.append(sample)
        if (i+1) % 50 == 0:
            logging.info(f"{i + 1} samples generated.")
    try:
        with open(OUTPUT_FILE, 'w', encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logging.info(f"Dataset generation complete. {NUM_SAMPLES} samples saved to {OUTPUT_FILE}.")
    except IOError as e:
        logging.error(f"Failed to write to file {OUTPUT_FILE}: {e}")


if __name__ == "__main__":
    main()