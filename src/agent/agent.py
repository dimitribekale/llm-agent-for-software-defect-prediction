import os
import re
import json
import pydoc
import requests
import tempfile
import subprocess
from prompts import SYSTEM_PROMPT
from typing import Any, Dict, List, Optional
from tools.websearch_tool import WebSearchTool
from tools.documentation_search_tool import DocumentationSearchTool




OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "codellama:7b"

def query_ollama(prompt, model=OLLAMA_MODEL, temperature=0.1, max_tokens=1024):
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]

def run_semgrep_on_code(code: str, language: str = "python") -> List[Dict[str, Any]]:
    # Use a temporary file for analysis
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



class DocumentationSearchTool:
    def __init__(self):
        # Mapping for official docs
        self.official_docs = {
            "python": "https://docs.python.org/3/library/",
            "java": "https://docs.oracle.com/javase/8/docs/api/",
            "c++": "https://en.cppreference.com/w/",
            "rust": "https://doc.rust-lang.org/std/",
            "c": "https://en.cppreference.com/w/c"
        }

    def optimize_query(self, method, language):
        return method.strip(), language.lower().strip()

    def search_python(self, method):
        # Use pydoc to get documentation for Python methods/classes
        try:
            doc = pydoc.render_doc(method, "Help on %s")
            return doc
        except Exception as e:
            return f"Could not find documentation for {method}: {e}"

    def search_other(self, method, language):
        # For other languages, provide a link to the official docs
        doc_link = self.official_docs.get(language)
        if doc_link:
            return f"Refer to the official {language.capitalize()} documentation for `{method}`:\n{doc_link}"
        else:
            return f"No documentation source configured for language: {language}"

    def __call__(self, method, language):
        print(f"[DocumentationSearchTool] Searching for: {method} in {language}")
        method, language = self.optimize_query(method, language)
        if language == "python":
            doc = self.search_python(method)
        else:
            doc = self.search_other(method, language)
        return doc


class DefectPredictionAgent:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
        self.history = []
        self.web_tool = WebSearchTool()
        self.doc_tool = DocumentationSearchTool()

    def loop(
        self,
        code_snippet: str,
        language: str = "python",
        code_metrics: Optional[Dict[str, Any]] = None,
        historical_defects: Optional[Any] = None
    ) -> str:
        # 1. Thought
        thought = self.think(code_snippet, language, code_metrics, historical_defects)
        print("\n[Thought]\n", thought)

        # 2. Action: Static analysis
        findings = run_semgrep_on_code(code_snippet, language)
        print("\n[Action]\nStatic analysis findings collected.")

        # 2b. Action: Documentation search (optional, e.g., for key methods)
        doc_info = ""
        # Try to extract function/method names for doc search (simple heuristic)
        matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_snippet)
        if matches:
            # Only search for the first function/method found
            doc_info = self.doc_tool(matches[0], language)
            print(f"\n[Action]\nDocumentation info for `{matches[0]}` collected.")

        # 3. Observation
        observation = self.observe(findings, doc_info)
        print("\n[Observation]\n", observation)

        # 4. Answer (LLM synthesis)
        answer = self.answer(
            code_snippet, language, thought, findings, doc_info, observation, code_metrics, historical_defects
        )
        print("\n[Answer]\n", answer)
        return answer

    def think(self, code_snippet, language, code_metrics, historical_defects):
        prompt = (
            f"You are preparing to predict software defects for the following code snippet.\n"
            f"Language: {language}\n"
            f"Code:\n{code_snippet}\n"
            f"Code metrics: {code_metrics if code_metrics else 'None'}\n"
            f"Historical defects: {historical_defects if historical_defects else 'None'}\n"
            f"Describe your initial thoughts, key risks, and what actions you will take."
        )
        full_prompt = self.system_prompt + "\n" + prompt
        return query_ollama(full_prompt)

    def observe(self, findings, doc_info):
        obs = format_semgrep_findings(findings)
        if doc_info:
            obs += "\n\nDocumentation Info:\n" + doc_info
        return obs

    def answer(
        self,
        code_snippet,
        language,
        thought,
        findings,
        doc_info,
        observation,
        code_metrics,
        historical_defects
    ):
        prompt = f"""
            You are a software defect prediction agent. 
            Here is the system prompt describing your responsibilities:
            {self.system_prompt}

            Loop summary:
            Thought: {thought}
            Static Analysis Findings: {findings}
            Documentation Info: {doc_info}
            Observation: {observation}

            Based on the above, provide:
            - A defect probability (if applicable)
            - A summary of defect-prone areas
            - Actionable recommendations for the developer
            - A clear explanation of your reasoning and process
            - (If relevant) Comments on specific lines of code that may induce defects

            Format your answer for a developer audience.
            """     
        full_prompt = self.system_prompt + "\n" + prompt
        return query_ollama(full_prompt)

# === Example Usage ===

if __name__ == "__main__":
    agent = DefectPredictionAgent(SYSTEM_PROMPT)
    # Example code snippet (Python)
    code = """
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)

data = [10, 20, 30, 40, 50]
print("Average:", calculate_average(data))
print("Average of empty list:", calculate_average([]))

"""
    result = agent.loop(code_snippet=code, language="python")
    print("\nFinal Prediction:\n", result)


