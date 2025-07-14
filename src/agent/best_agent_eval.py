import os
import re
import json
import tempfile
import requests
import subprocess
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from tools.websearch_tool import WebSearchTool
from tools.documentation_search_tool import DocumentationSearchTool
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, recall_score

system_prompt = """

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

3. Static Analysis:
You will perform static analysis on code snippets to identify common patterns.
You will use tools like Semgrep to analyze code for potential defects.

4. Documentation Search:
You will search for relevant documentation, code comments, and external resources
to gather context and insights about the code being analyzed.

5. Web Search:
You will perform web searches to gather up-to-date information about software defects, best practices in defect prevention, and relevant tools or frameworks.


Constraints
You will ensure maximum efficiency in your reasoning process by carefully following the constraints below:
- Be transparent about the assumptions and limitations of your predictions.
- Do not introduce biases into predictions; rely solely on data and objective metrics.


You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop, you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the available actions - then return PAUSE.
Observation will be the result of running those actions.




Your available actions are:

Action1 - Web Search:

This Action is designed to enable you to gather up-to-date, relevant,
and accurate background information from the web about software defects, bugs, software metrics
descriptions, and related topics. This Action helps you provide more informed and contextually
accurate responses by leveraging external knowledge that may not be readily available in
your training data.

Use Cases:
The `Web Search` Action should be used in the following scenarios:
a. Collect Software Defect Background Information:
   - To retrieve general explanations, examples, or causes of software defects and bugs.
   - To understand industry best practices for defect management and resolution.

b. Collect Software Metrics Descriptions:
   - To find definitions and detailed descriptions of software metrics such as cyclomatic complexity,
    code churn, lines of code (LOC), etc.
   - To discover how specific software metrics are calculated, interpreted, and applied in defect
    prediction or software quality analysis.

c. Collect Recent Trends and Updates:
   - To identify current trends, tools, and techniques related to software defect prediction,
   testing, or quality assurance.
   - To gather information about new research, algorithms, or frameworks in
   software metrics and defect prediction.

d. Clarify Unfamiliar Terms or Concepts:
   - To clarify or explain terms, methodologies, or concepts that you cannot fully resolve using
    your internal knowledge.

e. Retrieve User-Requested Web Searches:
   - To answer user queries explicitly asking for information from external online sources about
   software defects, metrics, or related topics.

When Not to Use this Action:
- Do not use the `Web Search` Action for questions that can be answered using your internal knowledge
or existing repository data.
- Avoid using this Action for highly specific questions that require insight into private repositories
or internal systems, as the web search cannot access such data.
- Refrain from using this Action for general programming assistance unless the user explicitly
requests recent or external references.

Input and Output
- This Action take as input a concise, well-formed query summarizing the specific information
you need to retrieve. You must optimize the query for effective web search results.
- The output of this Action will be a detailed response containing relevant information retrieved
from the web. This may include:
  - Definitions, descriptions, or examples.
  - Summaries of articles, blog posts, or documentation.
  - Links to authoritative sources where the user can learn more.

Usage Guidelines
a. Optimize the queries:
   - Rewrite user queries to ensure they are concise and focused on the topic of interest.
   - Use technical terms and keywords specific to software defects, bugs, or metrics
   to improve search relevance.

b. Prioritize Relevancy:
   - Focus on retrieving authoritative, high-quality information from trusted sources such as
   technical blogs, research papers, documentation, and industry-standard websites.

c. Citations and Transparency:
   - Provide proper citations for the information retrieved, including links to the original sources.
   - Be transparent about the reliability and origin of the information.

d. Iterative Improvement:
   - Refine the web search query if the initial results are not satisfactory.
   - Use context from the discussion to improve the focus of subsequent searches.

d. Respect Freshness:
   - Use recent information when the user asks for current trends, tools, or updates

Action2 - Documention Search:

This Action is designed to enable you to gather detailed information about programming methods,
functions, parameters, arguments, return types, and other relevant constructs from official
and widely-used documentation sources for Python, Java, C++, C, and Rust. The primary goal of
this Action is to enhance your ability to analyze and predict software defects by providing context
and understanding of how specific methods and constructs are used.

Use Cases:
This Action should be used in the following scenarios:
a. To Understand Methods and Functions:
   - To retrieve descriptions of methods or functions and their associated use cases.
   - To understand the expected input parameters, arguments, and return types.

b. Analyze Code Semantics:
   - To fetch semantic details that explain how a particular method or function operates,
   its constraints, or its side effects.
   - To identify common pitfalls or misuse patterns.

c. Improve Defect Predictions:
   - To gather insights into method behaviors that are often linked to defects, such as improper
   argument handling or edge cases.
   - To collect information that can help identify potential defects in code using the retrieved
   methods or constructs.

d. Explore Language-Specific Features:
   - To explore language-specific constructs, such as memory management in C++, ownership
   and borrowing in Rust, or type hints in Python, that may affect defect prediction.

e. User-Requested Searches:
   - To respond to user queries explicitly asking for information on specific methods, functions,
   or constructs in the supported programming languages.

When Not to Use:
- Do not use the `Documentation Search` Action for general programming help or theory that does
not involve specific methods, functions, or constructs.
- Avoid using the tool for unsupported languages or non-programming-related queries.

Input and Output:
- Input: A focused query specifying the method, function, parameter, return type,
or construct to search for, along with the programming language (e.g., "fetch Python `dict.get` method and its parameter descriptions").
- Output: Comprehensive, detailed information about the queried item, including:
  - Method or function descriptions.
  - Parameter and argument details.
  - Return type explanations.
  - Any additional helpful notes or examples.

Usage Guidelines:
a. Query Optimization:
   - Formulate precise and unambiguous queries, specifying the programming language and the method
   or construct of interest.
   - Include additional context, such as use cases or problem areas, when available.

b. Prioritize Authoritative Sources:
   - Fetch documentation from official or widely recognized sources, such as Python's official
   documentation, Java's Oracle docs, Rust's docs.rs, or C++ references.

c. Detailed Insights:
   - Retrieve and present detailed information that can directly aid in predicting or analyzing
   software defects.

d. Relevance and Accuracy:
   - Ensure the retrieved information is relevant to the query and accurately represents the
   method or construct as described in the documentation.
 """

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODELS_TO_EVALUATE = [
    "deepseek-r1:7b"
    #"codegemma:7b",
    #"deepseek-coder:6.7b",
    #"codellama:7b",
    #"qwen2.5-coder:7b"
]

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
            ["semgrep", "--json", tmp_path],
            capture_output=True, text=True
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
        self.web_tool = WebSearchTool()
        self.doc_tool = DocumentationSearchTool()

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
            (r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3')  # Quote unquoted keys
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
    

    def generate_search_query(self, code: str, findings: List[Dict]) -> str:
        """Generate search query based on code and analysis findings"""
        prompt = f"""
Analyze this code and static analysis findings to create a web search query 
for defect prediction research:

Code:
{code}

Findings:
{format_semgrep_findings(findings)}

Create a concise, focused search query that would help find relevant information 
about potential defects. Focus on specific code patterns and error types.

Respond ONLY with the search query in quotes. Example: "Python list index error prevention"
"""
        response = query_ollama(prompt, model=self.model_name)
        # Extract quoted query if present
        match = re.search(r'"([^"]+)"', response)
        return match.group(1) if match else response.strip()
    

    def loop(
        self,
        code_snippet: str,
        language: str = "python",
    ) -> Tuple[int, str]:
        
        max_steps = 5  # Prevent infinite loops
        steps = 0
        done = False

        while not done and steps < max_steps:
            # 1. Thought
            thought = self.think(code_snippet, language)
            print("\n[Thought]\nCurrently thinking about the code snippet...\n")

            # 2. Action: Static analysis
            findings = run_semgrep_on_code(code_snippet, language)
            print("\n[Action]\nStatic analysis findings collected.")

            # 2b. Action: Documentation search (optional)
            doc_info = ""
            matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_snippet)
            if matches:
                doc_info = self.doc_tool(matches[0], language)
                print(f"\n[Action]\nDocumentation info collected.")

            # 2c. Action: Generate and perform web search
            web_info = ""
            if findings:
                try:
                   search_query = self.generate_search_query(code_snippet, findings)
                   print(f"\n[Generated Search Query] {search_query}")
                   web_info, _ = self.web_tool(search_query)
                   print(f"\n[Web Search Results]\n{web_info}")
                except Exception as e:
                   print(f"Web search failed: {e}")

            # 3. Observation
            observation = self.observe(findings, doc_info, web_info)
            print("\n[Observation]\nObservation collected:\n", observation)

            # 4. Answer (LLM synthesis)
            answer = self.answer(
                code_snippet, language, thought, findings, doc_info, observation
            )
            print("\n[Answer]\n", answer)

            # === NEW: Terminate after one loop (since all actions are in one pass) ===
            done = True
            return self.parse_prediction(answer), answer  # (prediction, raw_response)

        if not done:
            print("[Info] Reached maximum steps without final answer.")

    def think(self, code_snippet, language):
        prompt = (
            f"You are preparing to predict software defects for the following code snippet.\n"
            f"Language: {language}\n"
            f"Code:\n{code_snippet}\n"
            f"Describe your initial thoughts, key risks, and what actions you will take."
        )
        # FIX: Prepend system prompt to user prompt
        full_prompt = self.system_prompt + "\n" + prompt
        return query_ollama(full_prompt, model=self.model_name)

    def observe(self, findings, doc_info, web_info):
       obs = format_semgrep_findings(findings)
       if doc_info:
           obs += "\n\nDocumentation Info:\n" + doc_info
       if web_info:
           obs += "\n\nWeb Search Results:\n" + web_info
       return obs


    def answer(
        self,
        code_snippet,
        language,
        thought,
        findings,
        doc_info,
        observation
    ):
        prompt = f"""
You are a software defect prediction classifier agent.
Here is the system prompt describing your responsibilities:
{self.system_prompt}

Analyze the provided code snippet and predict defects using this format:

Loop summary:
Thought: {thought}
Static Analysis Findings: {findings}
Documentation Info: {doc_info}
Observation: {observation}

Based on the above. You must respond with:
- 1 if code contains potential defects
- 0 if code appears correct

Respond ONLY with the A JSON format. No extra text.

Example of output:

Your output: {{
            "prediction": 1,
            "explanation": "YOUR RATIONALE HERE"
        }}

    """
       # FIX: Prepend system prompt to prompt
        full_prompt = self.system_prompt + "\n" + prompt
        return query_ollama(full_prompt, model=self.model_name)
    
def evaluate_agent(
    agent: DefectPredictionAgent,
    dataset_path: str
) -> Dict[str, float]:
    """Runs evaluation on full dataset"""
    df = pd.read_csv(dataset_path)
    #y_true = df['target'].values
    y_true = df['target'].values[:2]  # Limit to first 2 samples for testing
    y_pred = []

    #for _, row in tqdm(df.iterrows(), total=len(df), desc="Samples", leave=False):
    for _, row in tqdm(df[:2].iterrows(), total=2, desc="Samples", leave=False):
        code = row['code_samples']
        pred, _ = agent.loop(code_snippet=code)
        y_pred.append(pred)
        print(f"Predictions: {y_pred}")

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
            system_prompt=system_prompt,
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


# === Example Usage ===

if __name__ == "__main__":

    comparison_df = evaluate_models("transformed_data.csv")
    print("\nModel Comparison Results:")
    print(comparison_df.to_markdown(floatfmt=".2f"))


    """agent = DefectPredictionAgent(system_prompt)
    # Example code snippet (Python)
    metrics = evaluate_agent(agent, "transformed_data.csv")

    print(f"Defect Detection Metrics:\n"
      f"Accuracy: {metrics['accuracy']:.2f}\n"
      f"F1-Score: {metrics['f1']:.2f}\n"
      f"Precision: {metrics['precision']:.2f}\n"
      f"Recall: {metrics['recall']:.2f}\n"
      f"Classification Report:\n{metrics['report']}\n")"""
