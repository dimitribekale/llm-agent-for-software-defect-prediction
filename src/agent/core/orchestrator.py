import re
from typing import Tuple, Optional, List
from ..config import SystemConfig
from ..llm import (
    LLMClientFactory,
    PromptBuilder,
    PromptTemplates,
    LLMException
)
from ..tools import ToolRegistry
from ..parsers import DefectPredictionParser
from .state import (
    AgentState,
    AgentContext,
    MessageRole,
    Observation,
    ToolCall,
    DefectPrediction
)
from .memory import MemoryManager


class DefectPredictionOrchestrator:
    """
    Main orchestrator for defect prediction agent.

    Coordinates LLM, tools, memory, and state management in a
    structured Think-Act-Observe-Decide loop.
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize orchestrator.

        Args:
            config: System configuration (uses defaults if None)
            system_prompt: Custom system prompt (uses template if None)
        """
        self.config = config or SystemConfig.default()

        # Initialize components
        self.llm_client = LLMClientFactory.create("ollama", self.config.llm)
        self.tool_registry = ToolRegistry()
        self.memory = MemoryManager(self.config.memory)
        self.prompt_builder = PromptBuilder()

        # Set system prompt
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            tool_desc = self.tool_registry.format_tools_for_prompt()
            self.system_prompt = PromptTemplates.get_system_prompt(
                include_tools=len(self.tool_registry) > 0,
                tool_descriptions=tool_desc
            )

        self.memory.set_system_prompt(self.system_prompt)

        # State
        self.context: Optional[AgentContext] = None

    def register_tool(self, tool):
        """Register a tool with the agent."""
        self.tool_registry.register(tool)
        # Update system prompt with new tool
        tool_desc = self.tool_registry.format_tools_for_prompt()
        self.system_prompt = PromptTemplates.get_system_prompt(
            include_tools=True,
            tool_descriptions=tool_desc
        )
        self.memory.set_system_prompt(self.system_prompt)

    def predict(
        self,
        code_snippet: str,
        language: str = "python"
    ) -> Tuple[int, DefectPrediction]:
        """
        Main entry point for defect prediction.

        Args:
            code_snippet: Code to analyze
            language: Programming language

        Returns:
            Tuple of (prediction, full_prediction_object)
        """
        # Initialize context
        self.context = AgentContext(
            code_snippet=code_snippet,
            language=language,
            system_prompt=self.system_prompt
        )
        self.memory.clear()
        self.memory.set_system_prompt(self.system_prompt)

        # Log initial user request to memory
        self.memory.add_message(
            MessageRole.USER,
            f"Analyze this {language} code for defects:\n{code_snippet[:500]}..."
        )

        if self.config.agent.verbose:
            print(f"\n{'='*60}")
            print(f"Starting defect prediction for {language} code")
            print(f"{'='*60}")

        # Main loop
        try:
            prediction = self._execution_loop()
            self.context.complete(prediction)
            return prediction.prediction, prediction

        except Exception as e:
            if self.config.agent.verbose:
                print(f"\n[ERROR] Prediction failed: {str(e)}")

            # Return safe default
            error_prediction = DefectPrediction(
                prediction=0,
                confidence=0.0,
                explanation=f"Error during prediction: {str(e)}"
            )
            self.context.complete(error_prediction)
            return 0, error_prediction

    def _execution_loop(self) -> DefectPrediction:
        """
        Main Think-Act-Observe-Decide loop.

        Returns:
            Final defect prediction
        """
        max_iterations = self.config.agent.max_iterations

        for iteration in range(max_iterations):
            self.context.iteration = iteration

            if self.config.agent.verbose:
                print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            # 1. THINK: Analyze current state
            self.context.set_state(AgentState.THINKING)
            thoughts = self._think()

            # 2. ACT: Execute tools if needed
            self.context.set_state(AgentState.ACTING)
            tool_results = self._act(thoughts)

            # 3. OBSERVE: Synthesize findings
            self.context.set_state(AgentState.OBSERVING)
            observation = self._observe(thoughts, tool_results)
            self.context.add_observation(observation)
            self.memory.add_observation(observation)

            # 4. DECIDE: Check if ready for final answer
            self.context.set_state(AgentState.DECIDING)
            if self._should_finalize(observation):
                if self.config.agent.verbose:
                    print("\n[DECISION] Sufficient confidence reached. Finalizing...")
                break

        # Generate final prediction
        return self._finalize()

    def _think(self) -> str:
        """
        Generate thoughts about the code.

        Returns:
            Analysis thoughts as string
        """
        if self.config.agent.verbose:
            print("\n[THINKING] Analyzing code...")

        # Build prompt
        prompt = self.prompt_builder.build_thinking_prompt(
            code=self.context.code_snippet,
            language=self.context.language,
            iteration=self.context.iteration,
            previous_observations=self.context.observations
        )

        # Query LLM
        try:
            response = self.llm_client.generate(prompt)
            thoughts = response.content

            # Log thoughts to memory
            self.memory.add_message(
                MessageRole.ASSISTANT,
                thoughts,
                metadata={"phase": "thinking", "iteration": self.context.iteration}
            )

            if self.config.agent.verbose:
                print(f"  Thoughts: {thoughts[:200]}...")

            return thoughts

        except LLMException as e:
            if self.config.agent.verbose:
                print(f"  [ERROR] LLM failed: {str(e)}")
            return f"Error generating thoughts: {str(e)}"

    def _act(self, thoughts: str) -> List:
        """
        Execute tools based on thoughts.

        Args:
            thoughts: Analysis thoughts

        Returns:
            List of tool results
        """
        if len(self.tool_registry) == 0:
            return []

        if self.config.agent.verbose:
            print("\n[ACTING] Executing tools...")

        tool_calls = self._plan_tool_calls(thoughts)

        if not tool_calls:
            if self.config.agent.verbose:
                print("  No tools to execute")
            return []

        # Execute tools
        results = self.tool_registry.execute_multiple(
            tool_calls,
            parallel=self.config.agent.parallel_tools
        )

        # Log tool results to memory
        if results:
            tool_summary = "\n".join([
                f"{r.tool_name}: {'Success' if r.success else 'Failed'}"
                for r in results
            ])
            self.memory.add_message(
                MessageRole.TOOL,
                tool_summary,
                metadata={"tool_count": len(results)}
            )

        if self.config.agent.verbose:
            for result in results:
                status = "✓" if result.success else "✗"
                print(f"  {status} {result.tool_name}: {str(result.result)[:100]}...")

        return results

    def _plan_tool_calls(self, thoughts: str) -> List[ToolCall]:
        """
        Plan which tools to execute based on thoughts.

        Args:
            thoughts: Current analysis thoughts

        Returns:
            List of tool calls
        """
        tool_calls = []

        # Web search if enabled
        if (self.config.agent.enable_web_search and
            "web_search" in self.tool_registry):
            query = self._generate_search_query(thoughts)
            if query:
                tool_calls.append(ToolCall(
                    tool_name="web_search",
                    arguments={"query": query}
                ))

        # Documentation search if enabled
        if (self.config.agent.enable_doc_search and
            "documentation_search" in self.tool_registry):
            # Extract function names from code
            functions = self._extract_function_names(self.context.code_snippet)
            if functions:
                tool_calls.append(ToolCall(
                    tool_name="documentation_search",
                    arguments={
                        "method_name": functions[0],
                        "language": self.context.language
                    }
                ))

        return tool_calls

    def _generate_search_query(self, thoughts: str) -> Optional[str]:
        """Generate web search query from thoughts."""
        try:
            prompt = self.prompt_builder.build_search_query_prompt(
                code=self.context.code_snippet,
                thoughts=thoughts,
                language=self.context.language
            )
            response = self.llm_client.generate(prompt)
            # Extract query from quotes
            match = re.search(r'"([^"]+)"', response.content)
            return match.group(1) if match else response.content.strip()
        except Exception as e:
            return None

    def _extract_function_names(self, code: str) -> List[str]:
        """Extract function/method names from code."""
        matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        return list(set(matches))[:3]  # Return up to 3 unique names

    def _observe(self, thoughts: str, tool_results: List) -> Observation:
        """
        Synthesize thoughts and tool results into observation.

        Args:
            thoughts: Analysis thoughts
            tool_results: Tool execution results

        Returns:
            Observation object
        """
        if self.config.agent.verbose:
            print("\n[OBSERVING] Synthesizing findings...")

        observation = Observation(
            thoughts=thoughts,
            tool_results=tool_results,
            iteration=self.context.iteration
        )

        # Calculate confidence score
        observation.confidence_score = self._calculate_confidence(observation)

        return observation

    def _calculate_confidence(self, observation: Observation) -> float:
        """
        Calculate confidence score for observation.

        Args:
            observation: Observation to score

        Returns:
            Confidence score (0.0 to 1.0)
        """
        score = 0.0

        # Check for defect indicators
        text = observation.thoughts.lower()
        for pattern in self.config.agent.defect_keywords:
            if re.search(pattern, text):
                score += 0.2

        # Check for clean indicators
        for pattern in self.config.agent.clean_keywords:
            if re.search(pattern, text):
                score += 0.2

        # Successful tool results add confidence
        if observation.tool_results:
            success_ratio = sum(1 for r in observation.tool_results if r.success) / len(observation.tool_results)
            score += success_ratio * 0.3

        # Detailed observations add confidence
        if len(observation.thoughts.split('\n')) >= self.config.agent.min_observation_lines:
            score += 0.2

        return min(score, 1.0)

    def _should_finalize(self, observation: Observation) -> bool:
        """
        Determine if agent should finalize prediction.

        Args:
            observation: Latest observation

        Returns:
            True if should finalize
        """
        # Check confidence threshold
        if observation.confidence_score >= self.config.agent.confidence_threshold:
            return True

        # Check for definitive patterns
        text = observation.thoughts.lower()

        # Strong defect indicators
        if any(re.search(p, text) for p in self.config.agent.defect_keywords):
            return True

        # Strong clean indicators
        if any(re.search(p, text) for p in self.config.agent.clean_keywords):
            return True

        return False

    def _finalize(self) -> DefectPrediction:
        """
        Generate final defect prediction.

        Returns:
            DefectPrediction object
        """
        if self.config.agent.verbose:
            print("\n[FINALIZING] Generating prediction...")

        # Build final prompt
        prompt = self.prompt_builder.build_final_prediction_prompt(
            code=self.context.code_snippet,
            language=self.context.language,
            observations=self.context.observations
        )

        # Get prediction from LLM
        try:
            response = self.llm_client.generate(prompt)
            prediction = DefectPredictionParser.parse(response.content)

            # Log final prediction to memory
            self.memory.add_message(
                MessageRole.ASSISTANT,
                f"Final prediction: {prediction.prediction} (confidence: {prediction.confidence:.2f})\n"
                f"Explanation: {prediction.explanation}",
                metadata={"phase": "finalization", "prediction": prediction.prediction}
            )

            if self.config.agent.verbose:
                print(f"\n{'='*60}")
                print(f"PREDICTION: {prediction.prediction} (confidence: {prediction.confidence:.2f})")
                print(f"EXPLANATION: {prediction.explanation}")
                print(f"{'='*60}\n")

            return prediction

        except Exception as e:
            if self.config.agent.verbose:
                print(f"  [ERROR] Finalization failed: {str(e)}")

            return DefectPrediction(
                prediction=0,
                confidence=0.0,
                explanation=f"Error generating final prediction: {str(e)}"
            )

    def get_context(self) -> Optional[AgentContext]:
        """Get current execution context."""
        return self.context

    def get_statistics(self) -> dict:
        """Get execution statistics."""
        if not self.context:
            return {}

        return {
            "iterations": self.context.iteration + 1,
            "execution_time": self.context.get_execution_time(),
            "observations": len(self.context.observations),
            "state": self.context.current_state.name,
            "memory": self.memory.get_summary()
        }
