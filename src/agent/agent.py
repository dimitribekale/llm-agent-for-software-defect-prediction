import re
from utils import query_ollama
from typing import Dict, List, Tuple
from tools.websearch_tool import WebSearchTool
from tools.documentation_search_tool import DocumentationSearchTool




class DefectPredictionAgent:
    def __init__(self, system_prompt: str, model_name: str = "codegemma:7b"):

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.history = []
        self.web_tool = WebSearchTool()
        self.doc_tool = DocumentationSearchTool()
    

    def generate_search_query(self, code: str, thoughts: List[Dict]) -> str:
        """Generate search query based on code and analysis findings"""
        prompt = f"""
            Analyze this code and static analysis findings to create a web search query 
            for defect prediction research:

            Code:
            {code}

            Findings:
            {thoughts}

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
        
        max_loops = 5  # Prevent infinite loops
        history = []

        for step in range(max_loops):
            # 1. Thought
            thought = self.think(code_snippet, language)
            print("\n[Thought]\nCurrently thinking about the code snippet...\n")
            
            # 2b. Action: Documentation search (optional)
            doc_info = ""
            matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_snippet)
            if matches:
                doc_info = self.doc_tool(matches[0], language)
                print(f"\n[Action]\nDocumentation info collected.")

            # 2c. Action: Generate and perform web search
            web_info = ""
            if thought:
                try:
                   search_query = self.generate_search_query(code_snippet, thought)
                   print(f"\n[Generated Search Query] {search_query}")
                   web_info, _ = self.web_tool(search_query)
                   print(f"\n[Web Search Results]\n{web_info}")
                except Exception as e:
                   print(f"Web search failed: {e}")

            # 3. Observation
            observation = self.observe(thought, doc_info, web_info)
            print("\n[Observation]\nObservation collected:\n", observation)
            history.append(observation)

            # 4. Early exit if confident
            if self._is_confident(observation):
                return self._final_answer(code_snippet, thought, observation)

         # Final answer after max steps
        return self._final_answer(code_snippet, "Max analysis steps reached", observation)
    
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

    def observe(self, doc_info, web_info):
       obs = ""
       if doc_info:
           obs += "\n\nDocumentation Info:\n" + doc_info
       if web_info:
           obs += "\n\nWeb Search Results:\n" + web_info
       return obs


    def answer(
        self,
        thought: str,
        web_info: str,
        doc_info: str,
        observation: str
    ):
        prompt = f"""
           You are a software defect prediction classifier agent.
           Here is the system prompt describing your responsibilities:
           {self.system_prompt}

           Analyze the provided code snippet and predict defects using this format:

           Loop summary:
           Thought: {thought}
           Static Analysis Findings: {web_info}
           Documentation Info: {doc_info}
           Observation: {observation}

           """
       # FIX: Prepend system prompt to prompt
        full_prompt = self.system_prompt + "\n" + prompt
        return query_ollama(full_prompt, model=self.model_name)



class DefectPredictionAgentWithoutTools:
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
        self.max_loops = 5 
    
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
        
           # 3. Observe
           
           history.append(thought)
        
           # 4. Early exit if confident
           if self._is_confident(thought):
                return self._final_answer(code_snippet, thought)
    
        # Final answer after max steps
        return self._final_answer(code_snippet, "Max analysis steps reached", thought)

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
    
