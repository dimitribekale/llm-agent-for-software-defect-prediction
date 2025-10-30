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

        # Initialize components
        self.llm_client = LLMClientFactory.create("ollama", self.config.llm)
        self.intent_analyzer = CommitIntentAnalyzer(self.llm_client)
        self.prompt_builder = PromptBuilder()

        # Use commit-specific system prompt
        self.system_prompt = PromptTemplates.COMMIT_ANALYSIS_SYSTEM_PROMPT

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

        prediction = self._predict(prompt)

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

    def _predict(self, prompt: str) -> DefectPrediction:
        """
        Get defect prediction from LLM.

        Args:
            prompt: Analysis prompt

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
