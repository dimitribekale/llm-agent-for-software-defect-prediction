"""
Prompt building utilities for the defect prediction agent.
"""

from typing import Optional, List, Dict, Any
from ..core.state import Observation


class PromptBuilder:
    """Builder for constructing LLM prompts."""

    @staticmethod
    def build_thinking_prompt(
        code: str,
        language: str,
        iteration: int,
        previous_observations: Optional[List[Observation]] = None
    ) -> str:
        """
        Build prompt for thinking phase.

        Args:
            code: Code snippet to analyze
            language: Programming language
            iteration: Current iteration number
            previous_observations: Previous analysis observations

        Returns:
            Formatted thinking prompt
        """
        parts = [
            f"Analyze this {language} code for potential defects.",
            f"\nIteration: {iteration + 1}\n",
            "Code:",
            f"```{language}",
            code,
            "```\n"
        ]

        if previous_observations:
            parts.append("\nPrevious Analysis:")
            for obs in previous_observations[-2:]:  # Last 2
                parts.append(f"\n- Iteration {obs.iteration}: {obs.thoughts[:200]}...")

        parts.extend([
            "\nProvide your analysis focusing on:",
            "1. Potential bugs or errors",
            "2. Security vulnerabilities",
            "3. Code quality issues",
            "4. Edge cases that might fail",
            "\nWhat specific areas should we investigate further?"
        ])

        return "\n".join(parts)

    @staticmethod
    def build_final_prediction_prompt(
        code: str,
        language: str,
        observations: List[Observation],
        truncate_code: int = 2000
    ) -> str:
        """
        Build prompt for final prediction.

        Args:
            code: Code snippet
            language: Programming language
            observations: All observations collected
            truncate_code: Maximum code characters to include

        Returns:
            Formatted final prediction prompt
        """
        # Truncate code if too long
        if len(code) > truncate_code:
            code = code[:truncate_code] + "\n... (truncated)"

        parts = [
            f"Based on comprehensive analysis, predict if this {language} code has defects.",
            f"\nCode:",
            f"```{language}",
            code,
            "```\n",
            "\nAnalysis Summary:"
        ]

        # Add observation summaries
        for obs in observations:
            parts.append(f"\nIteration {obs.iteration}:")
            if obs.thoughts:
                parts.append(f"Thoughts: {obs.thoughts[:300]}")
            for tool_result in obs.tool_results:
                if tool_result.success:
                    result_str = str(tool_result.result)[:200]
                    parts.append(f"{tool_result.tool_name}: {result_str}")

        parts.extend([
            "\n\nWeigh all evidence and provide your final prediction.",
            "\nRespond with JSON:",
            "{",
            '  "prediction": 0 or 1,  // 0=clean, 1=defective',
            '  "confidence": 0.0-1.0,',
            '  "explanation": "brief rationale"',
            "}"
        ])

        return "\n".join(parts)

    @staticmethod
    def build_search_query_prompt(
        code: str,
        thoughts: str,
        language: str
    ) -> str:
        """
        Build prompt for generating web search query.

        Args:
            code: Code snippet
            thoughts: Current analysis thoughts
            language: Programming language

        Returns:
            Search query generation prompt
        """
        return f"""Based on this code analysis, generate a focused web search query to find relevant defect information.

Code ({language}):
{code[:500]}...

Analysis thoughts:
{thoughts[:300]}

Generate a concise search query (5-10 words) that would help find:
- Common defects in similar code patterns
- Security vulnerabilities
- Best practices violations

Respond with ONLY the search query in quotes, e.g., "Python list index error prevention"
"""

    @staticmethod
    def format_tool_results(
        results: List[Any],
        max_length: int = 1000
    ) -> str:
        """
        Format tool results for inclusion in prompts.

        Args:
            results: List of tool results
            max_length: Maximum length per result

        Returns:
            Formatted results string
        """
        if not results:
            return "No tool results available."

        parts = []
        for i, result in enumerate(results, 1):
            result_str = str(result)
            if len(result_str) > max_length:
                result_str = result_str[:max_length] + "...(truncated)"
            parts.append(f"{i}. {result_str}")

        return "\n".join(parts)

    @staticmethod
    def build_observation_summary(observation: Observation) -> str:
        """
        Build formatted summary of an observation.

        Args:
            observation: Observation to summarize

        Returns:
            Formatted summary string
        """
        parts = [f"Iteration {observation.iteration}:"]

        if observation.thoughts:
            parts.append(f"Analysis: {observation.thoughts[:200]}")

        if observation.tool_results:
            parts.append("\nTool Results:")
            for result in observation.tool_results:
                if result.success:
                    parts.append(f"  - {result.tool_name}: {str(result.result)[:150]}")
                else:
                    parts.append(f"  - {result.tool_name}: Failed ({result.error})")

        if observation.confidence_score > 0:
            parts.append(f"\nConfidence: {observation.confidence_score:.2f}")

        return "\n".join(parts)

    @staticmethod
    def build_commit_analysis_prompt(
        commit_hash: str,
        commit_message: str,
        intent_description: str,
        added_lines: List[str],
        removed_lines: List[str],
        file_names: List[str],
        stats: dict
    ) -> str:
        """
        Build prompt for analyzing a Git commit.

        Args:
            commit_hash: Git commit hash
            commit_message: Commit message
            intent_description: Analyzed developer intent
            added_lines: Lines of code added
            removed_lines: Lines of code removed
            file_names: Files changed
            stats: Commit statistics

        Returns:
            Formatted commit analysis prompt
        """
        # Limit lines to prevent context overflow
        added_sample = added_lines[:50] if len(added_lines) > 50 else added_lines
        removed_sample = removed_lines[:50] if len(removed_lines) > 50 else removed_lines

        parts = [
            f"Analyze this Git commit for potential defects.",
            f"\nCommit: {commit_hash[:8]}",
            f"\nCommit Message:",
            f'"""{commit_message}"""',
            f"\nDeveloper Intent:",
            f"{intent_description}",
            f"\nStatistics:",
            f"  - Files changed: {stats.get('files_changed', 0)}",
            f"  - Lines added: {stats.get('total_insertions', 0)}",
            f"  - Lines removed: {stats.get('total_deletions', 0)}",
            f"  - Net change: {stats.get('net_lines', 0)}",
            f"\nFiles Changed:",
        ]

        for file_name in file_names[:10]:  # Limit files shown
            parts.append(f"  - {file_name}")

        if len(file_names) > 10:
            parts.append(f"  ... and {len(file_names) - 10} more files")

        if removed_sample:
            parts.append(f"\nRemoved Code (- lines):")
            for line in removed_sample[:20]:
                if line.strip():
                    parts.append(f"  - {line}")
            if len(removed_lines) > 20:
                parts.append(f"  ... and {len(removed_lines) - 20} more removed lines")

        if added_sample:
            parts.append(f"\nAdded Code (+ lines):")
            for line in added_sample[:20]:
                if line.strip():
                    parts.append(f"  + {line}")
            if len(added_lines) > 20:
                parts.append(f"  ... and {len(added_lines) - 20} more added lines")

        parts.extend([
            "\nAnalyze the commit considering:",
            "1. Does the added code introduce bugs or vulnerabilities?",
            "2. Does removing code break existing functionality?",
            "3. Are there incomplete changes or missing error handling?",
            "4. Does this match the stated intent?",
            "5. Are there risky patterns (null checks, error handling, etc.)?",
            "\nRespond with JSON including prediction, confidence, explanation, defect_types, and risk_factors."
        ])

        return "\n".join(parts)


class PromptTemplates:
    """Collection of prompt templates."""

    SYSTEM_PROMPT_MINIMAL = """You are a software defect prediction expert.
Analyze code snippets and predict whether they contain defects that could cause malfunctions, crashes, or unpredictable behavior.

Respond with JSON:
{
  "prediction": 0 or 1,
  "confidence": 0.0-1.0,
  "explanation": "brief rationale"
}
"""

    COMMIT_ANALYSIS_SYSTEM_PROMPT = """You are a software defect prediction expert specializing in commit analysis.
Your goal is to predict whether a Git commit introduces defects by analyzing:
1. Code changes (added and removed lines)
2. Commit message and developer intent
3. Context and change patterns

Respond with JSON:
{
  "prediction": 0 or 1,
  "confidence": 0.0-1.0,
  "explanation": "brief rationale",
  "defect_types": ["list of potential defect types if any"],
  "risk_factors": ["list of identified risk factors"]
}
"""

    SYSTEM_PROMPT_WITH_REASONING = """You are an advanced software defect prediction expert.
Your goal is to identify potential defects in code that could cause malfunctions, crashes, or unpredictable behavior.

Analysis Process:
1. THINK: Analyze code structure, logic, and patterns
2. ACT: Use available tools to gather information
3. OBSERVE: Synthesize findings
4. DECIDE: Make final prediction with confidence

Focus on:
- Logic errors and edge cases
- Security vulnerabilities
- Resource management issues
- Type errors and null references
- Concurrency issues
- API misuse

Always respond with valid JSON:
{
  "prediction": 0 or 1,
  "confidence": 0.0-1.0,
  "explanation": "brief rationale"
}
"""

    @staticmethod
    def get_system_prompt(
        include_tools: bool = True,
        tool_descriptions: Optional[str] = None
    ) -> str:
        """
        Get system prompt with optional tool descriptions.

        Args:
            include_tools: Include tool information
            tool_descriptions: Formatted tool descriptions

        Returns:
            Complete system prompt
        """
        if include_tools and tool_descriptions:
            return f"""{PromptTemplates.SYSTEM_PROMPT_WITH_REASONING}

Available Tools:
{tool_descriptions}

Use tools strategically to gather evidence before making predictions.
"""
        return PromptTemplates.SYSTEM_PROMPT_WITH_REASONING
