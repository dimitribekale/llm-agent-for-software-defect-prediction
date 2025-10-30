"""
Robust JSON parser with multiple fallback strategies for handling LLM outputs.
"""

import json
import re
from typing import Any, Dict, Optional, Union
from ..core.state import DefectPrediction


class JSONParseException(Exception):
    """Raised when all JSON parsing strategies fail."""
    pass


class RobustJSONParser:
    """
    Parser with multiple strategies for extracting JSON from LLM responses.

    Strategies (in order):
    1. Direct JSON parsing
    2. Extract JSON from markdown code blocks
    3. Find JSON-like content between braces
    4. Regex-based field extraction
    5. Default/fallback values
    """

    @staticmethod
    def parse(
        text: str,
        expected_schema: Optional[Dict[str, type]] = None,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Parse JSON from text using multiple strategies.

        Args:
            text: Raw text potentially containing JSON
            expected_schema: Expected field types for validation
            strict: If True, raise exception on parse failure

        Returns:
            Parsed dictionary

        Raises:
            JSONParseException: If strict=True and all strategies fail
        """
        strategies = [
            RobustJSONParser._parse_direct,
            RobustJSONParser._parse_from_code_block,
            RobustJSONParser._parse_from_braces,
            RobustJSONParser._parse_with_regex,
        ]

        for strategy in strategies:
            try:
                result = strategy(text)
                if result:
                    # Validate against schema if provided
                    if expected_schema:
                        RobustJSONParser._validate_schema(result, expected_schema)
                    return result
            except Exception:
                continue

        # All strategies failed
        if strict:
            raise JSONParseException(f"Failed to parse JSON from: {text[:200]}...")

        # Return default structure
        return RobustJSONParser._get_default_structure(expected_schema)

    @staticmethod
    def _parse_direct(text: str) -> Optional[Dict[str, Any]]:
        """Strategy 1: Direct JSON parsing."""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse_from_code_block(text: str) -> Optional[Dict[str, Any]]:
        """Strategy 2: Extract from markdown code block."""
        # Match ```json ... ``` or ```{...}```
        patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(\{.*?\})\n```',
            r'```(.*?)```'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _parse_from_braces(text: str) -> Optional[Dict[str, Any]]:
        """Strategy 3: Find content between outermost braces."""
        # Find first { and last }
        start = text.find('{')
        end = text.rfind('}')

        if start != -1 and end != -1 and start < end:
            json_str = text[start:end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common issues
                json_str = RobustJSONParser._fix_common_json_issues(json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def _fix_common_json_issues(json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')

        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Fix unquoted keys
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)

        return json_str

    @staticmethod
    def _parse_with_regex(text: str) -> Optional[Dict[str, Any]]:
        """Strategy 4: Extract fields using regex patterns."""
        result = {}

        # Common field patterns
        patterns = {
            'prediction': [
                r'"prediction"\s*:\s*(\d+)',
                r'prediction:\s*(\d+)',
                r'"prediction":\s*"(\d+)"'
            ],
            'confidence': [
                r'"confidence"\s*:\s*([\d.]+)',
                r'confidence:\s*([\d.]+)'
            ],
            'explanation': [
                r'"explanation"\s*:\s*"([^"]+)"',
                r'explanation:\s*"([^"]+)"',
                r'"explanation"\s*:\s*\'([^\']+)\''
            ],
            'defect_types': [
                r'"defect_types"\s*:\s*\[(.*?)\]',
                r'defect_types:\s*\[(.*?)\]'
            ]
        }

        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    value = match.group(1)
                    # Type conversion
                    if field == 'prediction':
                        result[field] = int(value)
                    elif field == 'confidence':
                        result[field] = float(value)
                    elif field == 'defect_types':
                        # Parse array of strings
                        items = re.findall(r'"([^"]+)"', value)
                        result[field] = items
                    else:
                        result[field] = value
                    break

        return result if result else None

    @staticmethod
    def _validate_schema(data: Dict[str, Any], schema: Dict[str, type]) -> bool:
        """
        Validate data against expected schema.

        Args:
            data: Parsed data
            schema: Expected field types

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        for field, expected_type in schema.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    raise ValueError(
                        f"Field '{field}' has type {type(data[field])} "
                        f"but expected {expected_type}"
                    )
        return True

    @staticmethod
    def _get_default_structure(schema: Optional[Dict[str, type]] = None) -> Dict[str, Any]:
        """Get default structure based on schema."""
        if not schema:
            return {
                "prediction": 0,
                "confidence": 0.0,
                "explanation": "Failed to parse response"
            }

        defaults = {
            int: 0,
            float: 0.0,
            str: "",
            list: [],
            dict: {},
            bool: False
        }

        return {
            field: defaults.get(field_type, None)
            for field, field_type in schema.items()
        }


class DefectPredictionParser:
    """Specialized parser for defect predictions."""

    SCHEMA = {
        "prediction": int,
        "confidence": float,
        "explanation": str,
        "defect_types": list,
        "recommended_fix": str,
        "criticality": str
    }

    @staticmethod
    def parse(text: str, strict: bool = False) -> DefectPrediction:
        """
        Parse defect prediction from LLM output.

        Args:
            text: LLM response text
            strict: If True, raise exception on parse failure

        Returns:
            DefectPrediction object
        """
        try:
            # Try robust parsing
            data = RobustJSONParser.parse(text, strict=False)

            # Extract prediction (required)
            prediction = data.get("prediction")
            if prediction is None:
                # Try to infer from explanation
                prediction = DefectPredictionParser._infer_prediction(text)

            # Ensure prediction is 0 or 1
            if isinstance(prediction, str):
                prediction = 1 if prediction.lower() in ["1", "true", "yes", "defective"] else 0
            prediction = int(prediction) if prediction is not None else 0
            prediction = 1 if prediction > 0 else 0

            # Extract other fields with defaults
            confidence = float(data.get("confidence", 0.5))
            explanation = str(data.get("explanation", text[:200]))
            defect_types = data.get("defect_types", [])
            if isinstance(defect_types, str):
                defect_types = [defect_types]

            return DefectPrediction(
                prediction=prediction,
                confidence=confidence,
                explanation=explanation,
                defect_types=defect_types,
                recommended_fix=str(data.get("recommended_fix", "")),
                criticality=str(data.get("criticality", ""))
            )

        except Exception as e:
            if strict:
                raise JSONParseException(f"Failed to parse defect prediction: {str(e)}")

            # Return safe default
            return DefectPrediction(
                prediction=0,
                confidence=0.0,
                explanation=f"Parse error: {str(e)}. Raw: {text[:100]}",
                defect_types=[],
                recommended_fix="",
                criticality=""
            )

    @staticmethod
    def _infer_prediction(text: str) -> int:
        """Infer prediction from text content."""
        text_lower = text.lower()

        # Strong defect indicators
        defect_words = ['defect', 'bug', 'error', 'vulnerability', 'issue', 'problem', 'fail']
        clean_words = ['no defect', 'clean', 'safe', 'no issue', 'no problem']

        # Check clean indicators first (more specific)
        if any(word in text_lower for word in clean_words):
            return 0

        # Check defect indicators
        if any(word in text_lower for word in defect_words):
            return 1

        # Default to clean
        return 0
