import os
import re
import json
import tempfile
import requests
import subprocess
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, recall_score

SYSTEM_PROMPT = """

You are an advanced AI assistant specializing in software defect prediction. 
Your goal is to assist software developers, quality assurance engineers, and project managers
In identifying, predicting, and mitigating potential defects in software systems.
You will predict whether a code contains defects that could possibly cause a software to 
malfunction, crash, or exhibit unpredictable behaviors.

These are your key responsabilities:

1. Defect Prediction:
You will analyze code snippets to predict potential defects in the code.
You will spot parts of the code that are most likely to introduce bugs, errors,
or unpredictable behaviors.

2. Data analysis:
You will process and analyze software metrics (e.g., cyclomatic complexity, code churn,
commit history) to predict defect-prone areas of the code snippet.

Predict defects using:
1. Code structure analysis
2. Common error patterns
3. Language-specific best practices

Constraints
You will ensure maximum efficiency in your reasoning process by carefully following the constraints below:
- Be transparent about the assumptions and limitations of your predictions.
- Do not introduce biases into predictions; rely solely on data and objective metrics.


You run in a loop of Thought, PAUSE, Observation.
At the end of the loop, you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Observation will be the result of running those actions.

"""

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODELS_TO_EVALUATE = ["deepseek-r1:7b"]

""" 
    ==========================================================================
    ==========================================================================
                   Utility Functions for Ollama and Semgrep
    ==========================================================================
    ==========================================================================
    
"""

def query_ollama(prompt, model=None, temperature=0.1, max_tokens=1024):
    if model is None:
        raise ValueError("Model name must be specified")
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    # DEBUG: Print what is being sent
    #print("Sending to Ollama:", payload)
    response = requests.post(OLLAMA_API_URL, json=payload)
    #print("Ollama response code:", response.status_code)
    #print("Ollama response text:", response.text)
    response.raise_for_status()
    return response.json()["response"]

def run_semgrep_on_code(code: str, language: str = "python") -> List[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["semgrep", "--json", "--config=auto", "...", "--lang", "python", "-", tmp_path],
            input=code, capture_output=True, text=True
        )
        findings = json.loads(result.stdout)
        return findings.get("results", [])
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        os.remove(tmp_path)

def format_semgrep_findings(findings: List[Dict[str, Any]]) -> str:
    if not findings:
        return "No static analysis findings."
    return "\n".join(
        f"- {f.get('check_id', 'N/A')}: {f.get('extra', {}).get('message', '')} (line {f.get('start', {}).get('line', '?')})"
        for f in findings
    )

""" 
    ==========================================================================
    ==========================================================================
                   Defect Prediction Agent Implementation
    ==========================================================================
    ==========================================================================
    
"""

class DefectPredictionAgent:
    def __init__(self, system_prompt: str, model_name: str = "codegemma:7b"):

        self.model_name = model_name
        self.system_prompt = system_prompt + """
        You MUST respond with ONLY:
        {
            "prediction": 0 or 1,
            "explanation": "brief rationale"
        }
        """
        self.history = []
        self.max_loops = 1
        

    def parse_prediction(self, llm_response: str) -> int:
        """Enhanced prediction parser with multiple fallback strategies"""
        def extract_and_parse(pattern: str, content: str) -> Dict[str, Any]:
            """Helper: Extract JSON using regex and parse"""
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return json.loads(match.group(1).strip())
            raise json.JSONDecodeError("No match", "", 0)

        def validate_result(result: Dict[str, Any]) -> bool:
            """Validate parsed JSON structure"""
            return "prediction" in result and str(result["prediction"]) in ("0", "1")

        # Ordered parsing strategies
        strategies = [
            # Strategy 1: Direct JSON parse
            lambda: json.loads(llm_response),
            
            # Strategy 2: Extract JSON from ```
            lambda: extract_and_parse(r'```(?:json)?\n(.*?)(?=```)', llm_response),

            # Strategy 3: Extract from generic ``` blocks
            lambda: extract_and_parse(r'``````', llm_response),
            
            # Strategy 4: Find JSON-like structures
            lambda: self.find_json_in_text(llm_response),
            
            # Strategy 5: Sanitize and parse
            lambda: json.loads(self.sanitize_json(llm_response))
        ]

        # Try strategies in order
        for strategy in strategies:
            try:
                result = strategy()
                if validate_result(result):
                    return int(result["prediction"])
            except (json.JSONDecodeError, AttributeError, KeyError):
                continue

        # Final fallback: Keyword matching
        return self.pattern_fallback(llm_response)

    def find_json_in_text(self, content: str) -> Dict[str, Any]:
        """Find deepest valid JSON structure"""
        candidates = []
        stack = []
        start_idx = -1
        
        for i, char in enumerate(content):
            if char == '{':
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        candidates.append(content[start_idx:i+1])
        
        # Test candidates from longest to shortest
        for candidate in sorted(candidates, key=len, reverse=True):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise json.JSONDecodeError("No valid JSON", "", 0)

    def sanitize_json(self, content: str) -> str:
        """Fix common JSON issues"""
        fixes = [
            (r'(\w+):', r'"\1":'),  # Add quotes around keys
            (r"'(.*?)'", r'"\1"'),   # Replace single with double quotes
            (r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3'),  # Quote unquoted keys
            (r'(?<!\\)"', r'\"'),  # Escape unescaped double quotes
            (r'(\d+)(?!\w)', r'\1'),  # Ensure numbers are not followed by letters
        ]
        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content)
        return content

    def pattern_fallback(self, content: str) -> int:
        """Final keyword-based fallback"""
        content_lower = content.lower()
        defect_keywords = {"defect", "1", "yes", "error", "bug"}
        clean_keywords = {"clean", "0", "no", "correct"}
        return 1 if any(k in content_lower for k in defect_keywords) else 0
    
    def loop(
        self,
        code_snippet: str,
        language: str = "python",
    ) -> Tuple[int, str]:
        
        history = []

        for step in range(self.max_loops):
           # 1. Think (using history)
           thought = self._generate_thought(code_snippet, history)
           print("\n[Thought]\n")

           # 2. Act (static analysis)
           findings = run_semgrep_on_code(code_snippet, language)
           print("\n[Action]\nStatic analysis findings collected.")
        
           # 3. Observe
           observation = self._format_observation(findings)
           print("\n[Observation]\n")
           history.append(observation)
        
           # 4. Early exit if confident
           if self._is_confident(observation):
                return self._final_answer(code_snippet, thought, observation)
    
        # Final answer after max steps
        return self._final_answer(code_snippet, "Max analysis steps reached", observation)
    

    def _generate_thought(self, code: str, history: List[str]) -> str:
        """Generate analytical thoughts about potential defects"""
        prompt = f"""Analyze this code for defects considering previous observations:
           Code:
           {code}

           Previous Observations:
           {('\n'.join(history) or 'None')}

           Identify 2-3 key risk areas and analysis priorities."""
        return query_ollama(self.system_prompt + "\n" + prompt, self.model_name)
    
    
    def _format_observation(self, findings: List[dict]) -> str:
        """Format static analysis results"""
        errors = [f for f in findings if "error" in f]
        if errors:
           return f"Static analysis failed: {errors[0].get('error')}"
        if not findings:
            return "No static analysis findings."
        return "\n".join(
            f"{idx+1}. {f.get('check_id')}: {f.get('extra',{}).get('message')}"
            for idx, f in enumerate(findings)
        )

    def _final_answer(self, code: str, thought: str, observation: str) -> Tuple[int, str]:
        """Generate final prediction with chain-of-thought"""
        truncated_code = code[:2000] + "..." if len(code) > 2000 else code
        prompt = f"""Final defect prediction after analysis:

            Code:
            {truncated_code}

            Analysis Steps:
            {thought}

            Key Findings:
            {observation}

            Weigh the evidence and respond with JSON."""
        
        response = query_ollama(self.system_prompt + "\n" + prompt, self.model_name)
        return self.parse_prediction(response), response

    
    def _is_confident(self, observation: str) -> bool:
       """Determine confidence to finalize prediction based on observation content.
    
    Args:
        observation: Formatted string from static analysis findings
        
    Returns:
        bool: True if confident to finalize prediction, False otherwise
       """
       if "Static analysis failed" in observation:
          return False
       
       if not observation.strip():
          return False  # Empty observation
    
       # Normalize for case-insensitive matching
       obs_lower = observation.lower()
    
       # Strong defect indicators (regex whole-word matches)
       defect_patterns = [
           r'\berror\b', r'\bbug\b', r'\bvulnerab', 
           r'\bfail\b', r'\bexception\b', r'\bnull\b',
           r'\bindexerror\b', r'\btypeerror\b', r'\bcritical\b'
        ]
    
       # Clear non-defect indicators
       clean_patterns = [
           r'\bno issues?\b', r'\bno defects?\b', 
           r'\bclean\b', r'\bpassed\b', r'\bsafe\b'
       ]
    
       # Check for definitive patterns
       if any(re.search(pattern, obs_lower) for pattern in defect_patterns):
          return True
    
       if any(re.search(pattern, obs_lower) for pattern in clean_patterns):
          return True
    
       # Heuristic: Detailed findings likely contain actionable info
       if len(observation.split('\n')) > 2:  # At least 3 lines of findings
          return True
    
       # Confidence threshold not met
       return False
    

""" 
    ==========================================================================
    ==========================================================================
                   Defect Prediction Agent Evaluation Script
    ==========================================================================
    ==========================================================================
    
"""
    
def evaluate_agent(
    agent: DefectPredictionAgent,
    dataset_path: str
) -> Dict[str, float]:
    """Runs evaluation on full dataset"""


    df = pd.read_csv(dataset_path)

    y_true = df['target'].values
    y_pred = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating"):
        code = row['code_samples']
        pred, _ = agent.loop(code)
        y_pred.append(pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "report": classification_report(y_true, y_pred, target_names=['Clean', 'Defective'], zero_division=0)
    }

def evaluate_models(dataset_path: str) -> pd.DataFrame:
    """Evaluate all models and return comparison DataFrame"""
    results = []
    
    for model in tqdm(MODELS_TO_EVALUATE, desc="Evaluating Models"):
        # Initialize agent with current model
        agent = DefectPredictionAgent(
            system_prompt= SYSTEM_PROMPT,
            model_name=model  # Add this parameter to your __init__
        )
        
        # Run evaluation
        metrics = evaluate_agent(agent, dataset_path)
        metrics["model"] = model
        results.append(metrics)
    
    # Create formatted DataFrame
    df = pd.DataFrame(results).set_index("model")
    column_order = ["accuracy", "precision", "recall", "f1"]
    return df[column_order].sort_values("f1", ascending=False)





""" 
    ==========================================================================
    ==========================================================================
                   Usage Example for Defect Prediction Agent
    ==========================================================================
    ==========================================================================
    
"""

if __name__ == "__main__":

    comparison_df = evaluate_models("transformed_data.csv")
    print("\nModel Comparison Results:")
    print(comparison_df.to_markdown(floatfmt=".2f"))

