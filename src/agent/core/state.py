from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto
from datetime import datetime


class AgentState(Enum):
    """Agent execution states."""
    IDLE = auto()
    THINKING = auto()
    ACTING = auto()
    OBSERVING = auto()
    DECIDING = auto()
    COMPLETED = auto()
    ERROR = auto()


class MessageRole(Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Structured message in agent history."""
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __str__(self) -> str:
        return f"[{self.role.value}] {self.content[:100]}..."

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now().timestamp())
        )


@dataclass
class ToolCall:
    """Represents a tool invocation."""
    tool_name: str
    arguments: Dict[str, Any]
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "timestamp": self.timestamp
        }


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": str(self.result),
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp
        }


@dataclass
class Observation:
    """Agent observation combining multiple sources."""
    thoughts: str = ""
    tool_results: List[ToolResult] = field(default_factory=list)
    confidence_score: float = 0.0
    iteration: int = 0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def add_tool_result(self, result: ToolResult):
        """Add a tool result to observation."""
        self.tool_results.append(result)

    def has_errors(self) -> bool:
        """Check if any tool execution had errors."""
        return any(not r.success for r in self.tool_results)

    def format_summary(self) -> str:
        """Format observation as readable summary."""
        summary = []
        if self.thoughts:
            summary.append(f"Thoughts: {self.thoughts}")

        for result in self.tool_results:
            if result.success:
                summary.append(f"\n{result.tool_name}: {result.result}")
            else:
                summary.append(f"\n{result.tool_name} (failed): {result.error}")

        return "\n".join(summary)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thoughts": self.thoughts,
            "tool_results": [r.to_dict() for r in self.tool_results],
            "confidence_score": self.confidence_score,
            "iteration": self.iteration,
            "timestamp": self.timestamp
        }


@dataclass
class DefectPrediction:
    """Final defect prediction result."""
    prediction: int  # 0 = clean, 1 = defective
    confidence: float
    explanation: str
    defect_types: List[str] = field(default_factory=list)
    recommended_fix: str = ""
    criticality: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "defect_types": self.defect_types,
            "recommended_fix": self.recommended_fix,
            "criticality": self.criticality,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DefectPrediction":
        """Create prediction from dictionary."""
        return cls(
            prediction=data.get("prediction", 0),
            confidence=data.get("confidence", 0.0),
            explanation=data.get("explanation", ""),
            defect_types=data.get("defect_types", []),
            recommended_fix=data.get("recommended_fix", ""),
            criticality=data.get("criticality", ""),
            metadata=data.get("metadata", {})
        )


@dataclass
class AgentContext:
    """Complete agent execution context."""
    code_snippet: str
    language: str
    system_prompt: str
    current_state: AgentState = AgentState.IDLE
    iteration: int = 0
    observations: List[Observation] = field(default_factory=list)
    final_prediction: Optional[DefectPrediction] = None
    start_time: float = field(default_factory=lambda: datetime.now().timestamp())
    end_time: Optional[float] = None

    def add_observation(self, observation: Observation):
        """Add an observation to context."""
        observation.iteration = self.iteration
        self.observations.append(observation)

    def get_latest_observation(self) -> Optional[Observation]:
        """Get the most recent observation."""
        return self.observations[-1] if self.observations else None

    def set_state(self, state: AgentState):
        """Update agent state."""
        self.current_state = state

    def complete(self, prediction: DefectPrediction):
        """Mark context as completed."""
        self.final_prediction = prediction
        self.current_state = AgentState.COMPLETED
        self.end_time = datetime.now().timestamp()

    def get_execution_time(self) -> float:
        """Get total execution time in seconds."""
        end = self.end_time or datetime.now().timestamp()
        return end - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code_snippet": self.code_snippet,
            "language": self.language,
            "current_state": self.current_state.name,
            "iteration": self.iteration,
            "observations": [o.to_dict() for o in self.observations],
            "final_prediction": self.final_prediction.to_dict() if self.final_prediction else None,
            "execution_time": self.get_execution_time()
        }
