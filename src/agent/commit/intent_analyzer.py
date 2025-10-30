"""
Commit intent analyzer for understanding developer intentions from commit messages.
"""

from typing import Dict, Any, Optional
from ..llm import BaseLLMClient, LLMException
from ..parsers import RobustJSONParser


class CommitIntent:
    """Structured intent extracted from commit message."""

    def __init__(
        self,
        intent_type: str,
        description: str,
        is_bugfix: bool,
        is_feature: bool,
        is_refactor: bool,
        risk_level: str,
        confidence: float
    ):
        self.intent_type = intent_type
        self.description = description
        self.is_bugfix = is_bugfix
        self.is_feature = is_feature
        self.is_refactor = is_refactor
        self.risk_level = risk_level  # low, medium, high
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent_type": self.intent_type,
            "description": self.description,
            "is_bugfix": self.is_bugfix,
            "is_feature": self.is_feature,
            "is_refactor": self.is_refactor,
            "risk_level": self.risk_level,
            "confidence": self.confidence
        }

    def __repr__(self) -> str:
        return f"CommitIntent({self.intent_type}, risk={self.risk_level})"


class CommitIntentAnalyzer:
    """
    Analyzes commit messages to understand developer intent.

    Uses LLM to extract:
    - What the developer intended to do
    - Type of change (bugfix, feature, refactor, etc.)
    - Potential risk level
    """

    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize analyzer.

        Args:
            llm_client: LLM client for intent analysis
        """
        self.llm_client = llm_client

    def analyze(self, commit_message: str) -> CommitIntent:
        """
        Analyze commit message to extract developer intent.

        Args:
            commit_message: Git commit message

        Returns:
            CommitIntent object
        """
        prompt = self._build_intent_prompt(commit_message)

        try:
            response = self.llm_client.generate(prompt)
            intent_data = self._parse_intent_response(response.content)
            return self._create_intent(intent_data)

        except LLMException as e:
            # Return default intent if LLM fails
            return CommitIntent(
                intent_type="unknown",
                description=f"Failed to analyze: {str(e)}",
                is_bugfix=False,
                is_feature=False,
                is_refactor=False,
                risk_level="unknown",
                confidence=0.0
            )

    def _build_intent_prompt(self, commit_message: str) -> str:
        """
        Build prompt for intent analysis.

        Args:
            commit_message: Commit message

        Returns:
            Formatted prompt
        """
        return f"""Analyze this Git commit message and extract the developer's intent.

Commit Message:
\"\"\"{commit_message}\"\"\"

Classify the change and assess risk. Respond with JSON:
{{
    "intent_type": "one of: bugfix, feature, refactor, documentation, test, style, other",
    "description": "brief description of what developer intended to do",
    "is_bugfix": true or false,
    "is_feature": true or false,
    "is_refactor": true or false,
    "risk_level": "low, medium, or high",
    "confidence": 0.0 to 1.0
}}

Risk level guidelines:
- low: Documentation, tests, minor refactoring
- medium: Features, non-critical bugfixes
- high: Critical bugfixes, major refactors, architectural changes

Analyze carefully and respond with only the JSON."""

    def _parse_intent_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract intent data.

        Args:
            response: LLM response text

        Returns:
            Dictionary with intent data
        """
        # Use robust JSON parser
        try:
            data = RobustJSONParser.parse(response, strict=False)
            return data
        except Exception:
            # Fallback to default structure
            return {
                "intent_type": "unknown",
                "description": response[:200],
                "is_bugfix": False,
                "is_feature": False,
                "is_refactor": False,
                "risk_level": "unknown",
                "confidence": 0.0
            }

    def _create_intent(self, data: Dict[str, Any]) -> CommitIntent:
        """
        Create CommitIntent from parsed data.

        Args:
            data: Parsed intent data

        Returns:
            CommitIntent object
        """
        # Extract and validate fields
        intent_type = str(data.get("intent_type", "unknown"))
        description = str(data.get("description", ""))
        is_bugfix = bool(data.get("is_bugfix", False))
        is_feature = bool(data.get("is_feature", False))
        is_refactor = bool(data.get("is_refactor", False))
        risk_level = str(data.get("risk_level", "unknown"))
        confidence = float(data.get("confidence", 0.5))

        # Validate risk level
        valid_risk_levels = ["low", "medium", "high", "unknown"]
        if risk_level.lower() not in valid_risk_levels:
            risk_level = "unknown"

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        return CommitIntent(
            intent_type=intent_type,
            description=description,
            is_bugfix=is_bugfix,
            is_feature=is_feature,
            is_refactor=is_refactor,
            risk_level=risk_level,
            confidence=confidence
        )

    def analyze_batch(self, commit_messages: list[str]) -> list[CommitIntent]:
        """
        Analyze multiple commit messages.

        Args:
            commit_messages: List of commit messages

        Returns:
            List of CommitIntent objects
        """
        from tqdm import tqdm

        intents = []
        for message in tqdm(commit_messages, desc="Analyzing intents", unit="commit"):
            intent = self.analyze(message)
            intents.append(intent)

        return intents
