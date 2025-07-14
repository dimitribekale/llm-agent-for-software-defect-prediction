import re
import json
from typing import Any, Dict, List, Optional, Tuple
from tools.websearch_tool import WebSearchTool
from tools.documentation_search_tool import DocumentationSearchTool
from utils import (query_ollama,
                   format_semgrep_findings,
                   run_semgrep_on_code)





OLLAMA_MODEL = "codellama:7b"


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

# if __name__ == "__main__":
#     agent = DefectPredictionAgent(SYSTEM_PROMPT)
#     # Example code snippet (Python)
#     code = """
# def calculate_average(numbers):
#     total = 0
#     for n in numbers:
#         total += n
#     return total / len(numbers)

# data = [10, 20, 30, 40, 50]
# print("Average:", calculate_average(data))
# print("Average of empty list:", calculate_average([]))

# """
#     result = agent.loop(code_snippet=code, language="python")
#     print("\nFinal Prediction:\n", result)



class DefectPredictionAgent4:
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
    
