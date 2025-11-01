"""
Commit defect orchestrator for repository-based defect prediction.
"""

from typing import Tuple, Optional
from ..config import SystemConfig
from ..llm import LLMClientFactory, PromptBuilder, PromptTemplates
from ..parsers import DefectPredictionParser
from ..core.state import DefectPrediction
from ..tools.implementations import CommitData
from .intent_analyzer import CommitIntentAnalyzer, CommitIntent


class CommitDefectResult:
    """Result of commit defect prediction."""

    def __init__(
        self,
        commit_data: CommitData,
        intent: CommitIntent,
        prediction: DefectPrediction
    ):
        self.commit_data = commit_data
        self.intent = intent
        self.prediction = prediction

    @property
    def commit_hash(self) -> str:
        return self.commit_data.commit_hash

    @property
    def is_defective(self) -> bool:
        return self.prediction.prediction == 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "commit_hash": self.commit_data.commit_hash,
            "author": self.commit_data.author,
            "date": self.commit_data.date.isoformat(),
            "message": self.commit_data.message,
            "files_changed": self.commit_data.changed_files,
            "stats": self.commit_data.stats,
            "intent": self.intent.to_dict(),
            "prediction": self.prediction.to_dict()
        }

    def __repr__(self) -> str:
        status = "DEFECTIVE" if self.is_defective else "CLEAN"
        return f"CommitDefectResult({self.commit_hash[:8]}, {status})"


class CommitDefectOrchestrator:
    """
    Orchestrator for commit-based defect prediction.

    Analyzes Git commits by:
    1. Understanding developer intent from commit message
    2. Analyzing code changes (added/removed lines)
    3. Predicting if commit introduces defects
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        verbose: bool = True
    ):
        """
        Initialize orchestrator.

        Args:
            config: System configuration
            verbose: Print detailed output
        """
        self.config = config or SystemConfig.default()
        self.config.agent.verbose = verbose
        self.verbose = verbose
        self.tools = {}  # Registered tools

        # Initialize components
        self.llm_client = LLMClientFactory.create("ollama", self.config.llm)
        self.intent_analyzer = CommitIntentAnalyzer(self.llm_client)
        self.prompt_builder = PromptBuilder()

        # Use commit-specific system prompt
        self.system_prompt = PromptTemplates.COMMIT_ANALYSIS_SYSTEM_PROMPT

    def register_tool(self, tool):
        """
        Register a tool for use during analysis.

        Args:
            tool: Tool instance to register
        """
        tool_name = tool.get_metadata().name
        self.tools[tool_name] = tool
        if self.verbose:
            print(f"  Registered tool: {tool_name}")

    def predict_commit(self, commit_data: CommitData) -> CommitDefectResult:
        """
        Predict if a commit introduces defects.

        Args:
            commit_data: Commit data to analyze

        Returns:
            CommitDefectResult with prediction
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Analyzing commit: {commit_data.commit_hash[:8]}")
            print(f"Author: {commit_data.author}")
            print(f"Date: {commit_data.date}")
            print(f"Files: {len(commit_data.changed_files)}")
            print(f"{'='*70}")

        # Step 1: Analyze intent from commit message
        if self.verbose:
            print("\n[Step 1/3] Analyzing developer intent...")

        intent = self.intent_analyzer.analyze(commit_data.message)

        if self.verbose:
            print(f"  Intent: {intent.intent_type}")
            print(f"  Description: {intent.description}")
            print(f"  Risk Level: {intent.risk_level}")
            print(f"  Confidence: {intent.confidence:.2f}")

        # Step 2: Build analysis prompt
        if self.verbose:
            print("\n[Step 2/3] Building analysis prompt...")

        prompt = self._build_commit_prompt(commit_data, intent)

        # Step 3: Predict defects
        if self.verbose:
            print("\n[Step 3/3] Predicting defects...")

        prediction = self._predict(prompt, commit_data)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"PREDICTION: {prediction.prediction} ({'DEFECTIVE' if prediction.prediction == 1 else 'CLEAN'})")
            print(f"CONFIDENCE: {prediction.confidence:.2f}")
            print(f"EXPLANATION: {prediction.explanation}")
            if prediction.defect_types:
                print(f"DEFECT TYPES: {', '.join(prediction.defect_types)}")
            print(f"{'='*70}\n")

        return CommitDefectResult(
            commit_data=commit_data,
            intent=intent,
            prediction=prediction
        )

    def _build_commit_prompt(
        self,
        commit_data: CommitData,
        intent: CommitIntent
    ) -> str:
        """
        Build prompt for commit analysis.

        Args:
            commit_data: Commit data
            intent: Analyzed intent

        Returns:
            Formatted prompt
        """
        return self.prompt_builder.build_commit_analysis_prompt(
            commit_hash=commit_data.commit_hash,
            commit_message=commit_data.message,
            intent_description=intent.description,
            added_lines=commit_data.added_lines,
            removed_lines=commit_data.removed_lines,
            file_names=commit_data.changed_files,
            stats=commit_data.stats
        )

    def _predict(self, prompt: str, commit_data: CommitData = None) -> DefectPrediction:
        """
        Get defect prediction from LLM.

        Args:
            prompt: Analysis prompt
            commit_data: Optional commit data for tool usage analysis

        Returns:
            DefectPrediction
        """
        try:
            # Prepend system prompt
            full_prompt = self.system_prompt + "\n\n" + prompt

            # Query LLM
            response = self.llm_client.generate(full_prompt)

            # Parse prediction
            prediction = DefectPredictionParser.parse(response.content)

            # Check if we should use tools to verify uncertain predictions
            if self.tools and commit_data:
                prediction = self._verify_with_tools(prediction, commit_data)

            return prediction

        except Exception as e:
            if self.verbose:
                print(f"  [ERROR] Prediction failed: {str(e)}")

            # Return safe default
            return DefectPrediction(
                prediction=0,
                confidence=0.0,
                explanation=f"Error during prediction: {str(e)}"
            )

    def _verify_with_tools(self, prediction: DefectPrediction, commit_data: CommitData) -> DefectPrediction:
        """
        Use tools to verify uncertain predictions.

        Args:
            prediction: Initial prediction
            commit_data: Commit data

        Returns:
            Updated prediction with tool insights
        """
        # Decision: Use tools if confidence is low or code has import/package usage
        should_use_tools = (
            prediction.confidence < 0.75 or  # Low confidence
            self._has_package_usage(commit_data)  # Uses external packages
        )

        if not should_use_tools:
            return prediction

        if self.verbose:
            print(f"\n  🔍 Low confidence ({prediction.confidence:.2f}) or package usage detected")
            print(f"  Decision: Using external tools to verify prediction...")

        tool_insights = []

        # Use documentation search for package/API verification
        if "documentation_search" in self.tools:
            doc_insights = self._use_documentation_search(commit_data)
            if doc_insights:
                tool_insights.append(doc_insights)

        # Use web search for broader context
        if "web_search" in self.tools and prediction.confidence < 0.6:
            web_insights = self._use_web_search(commit_data, prediction)
            if web_insights:
                tool_insights.append(web_insights)

        # Update prediction with tool insights
        if tool_insights:
            updated_explanation = self._integrate_tool_insights(
                prediction.explanation,
                tool_insights
            )
            prediction.explanation = updated_explanation

        return prediction

    def _has_package_usage(self, commit_data: CommitData) -> bool:
        """Check if code uses external packages."""
        import_keywords = ['import ', 'from ', 'require(', 'include ', '#include']

        for line in commit_data.added_lines[:50]:  # Check first 50 lines
            if any(keyword in line for keyword in import_keywords):
                return True
        return False

    def _use_documentation_search(self, commit_data: CommitData) -> Optional[str]:
        """Use documentation search to verify package APIs."""
        tool = self.tools.get("documentation_search")
        if not tool:
            return None

        # Extract package names from imports
        packages = self._extract_packages(commit_data.added_lines)
        if not packages:
            return None

        for package in packages[:2]:  # Check top 2 packages
            query = f"{package} API documentation latest version"

            if self.verbose:
                print(f"\n  📚 Documentation Search:")
                print(f"     Query: \"{query}\"")

            try:
                result = tool.execute(query=query)

                if result and self.verbose:
                    summary = result[:200] + "..." if len(result) > 200 else result
                    print(f"     Summary: {summary}")

                return f"Documentation check for {package}: {result[:300]}"

            except Exception as e:
                if self.verbose:
                    print(f"     Error: {str(e)}")

        return None

    def _use_web_search(self, commit_data: CommitData, prediction: DefectPrediction) -> Optional[str]:
        """Use web search for additional context."""
        tool = self.tools.get("web_search")
        if not tool:
            return None

        # Build query based on defect types or code patterns
        if prediction.defect_types:
            query = f"{prediction.defect_types[0]} common causes and solutions"
        else:
            query = "common software defects in code changes"

        if self.verbose:
            print(f"\n  🌐 Web Search:")
            print(f"     Query: \"{query}\"")

        try:
            result = tool.execute(query=query)

            if result and self.verbose:
                summary = result[:200] + "..." if len(result) > 200 else result
                print(f"     Summary: {summary}")

            return f"Web search insights: {result[:300]}"

        except Exception as e:
            if self.verbose:
                print(f"     Error: {str(e)}")
            return None

    def _extract_packages(self, lines: list) -> list:
        """Extract package names from import statements."""
        packages = []
        import_patterns = [
            'import ',
            'from ',
        ]

        for line in lines[:50]:  # Check first 50 lines
            for pattern in import_patterns:
                if pattern in line:
                    # Simple extraction - get word after keyword
                    parts = line.split(pattern)
                    if len(parts) > 1:
                        pkg = parts[1].split()[0].split('.')[0].strip()
                        if pkg and pkg not in packages:
                            packages.append(pkg)

        return packages[:5]  # Return top 5

    def _integrate_tool_insights(self, original_explanation: str, insights: list) -> str:
        """Integrate tool insights into explanation."""
        insights_text = "\n\nVerified with external tools:\n" + "\n".join(f"- {insight}" for insight in insights)
        return original_explanation + insights_text
