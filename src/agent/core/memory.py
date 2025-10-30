from typing import List, Optional, Dict, Any
from collections import deque
from .state import Message, MessageRole, Observation
from ..config import MemoryConfig


class MemoryManager:
    """
    Manages agent's conversation history and context.

    Features:
    - Maintains message history with size limits
    - Supports different pruning strategies
    - Tracks observations across iterations
    - Provides context window management
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._messages: deque = deque(maxlen=self.config.max_history_size)
        self._observations: List[Observation] = []
        self._system_prompt: Optional[str] = None

    def set_system_prompt(self, prompt: str):
        """Set the system prompt."""
        self._system_prompt = prompt
        # Add as first message if not already present
        if not self._messages or self._messages[0].role != MessageRole.SYSTEM:
            self.add_message(MessageRole.SYSTEM, prompt)

    def add_message(self, role: MessageRole, content: str, metadata: Optional[Dict] = None):
        """
        Add a message to history.

        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content
            metadata: Optional metadata
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self._messages.append(message)

    def add_observation(self, observation: Observation):
        """Add an observation to memory."""
        self._observations.append(observation)

    def get_messages(self) -> List[Message]:
        """Get all messages in history."""
        return list(self._messages)

    def get_observations(self) -> List[Observation]:
        """Get all observations."""
        return self._observations

    def get_latest_observation(self) -> Optional[Observation]:
        """Get the most recent observation."""
        return self._observations[-1] if self._observations else None

    def get_context_for_prompt(
        self,
        include_observations: bool = True,
        max_chars: Optional[int] = None
    ) -> str:
        """
        Build context string for LLM prompt.

        Args:
            include_observations: Include observation summaries
            max_chars: Maximum characters (truncates if exceeded)

        Returns:
            Formatted context string
        """
        parts = []

        # Add message history
        for msg in self._messages:
            if msg.role == MessageRole.SYSTEM:
                continue  # System prompt handled separately
            parts.append(f"[{msg.role.value}] {msg.content}")

        # Add observations
        if include_observations and self._observations:
            parts.append("\n=== Previous Analysis ===")
            for i, obs in enumerate(self._observations[-3:], 1):  # Last 3
                parts.append(f"\nIteration {obs.iteration}:")
                parts.append(obs.format_summary())

        context = "\n".join(parts)

        # Truncate if needed
        if max_chars and len(context) > max_chars:
            context = context[-max_chars:]
            context = "...(truncated)...\n" + context

        return context

    def build_full_prompt(
        self,
        user_prompt: str,
        include_history: bool = True
    ) -> str:
        """
        Build complete prompt with system prompt, history, and user prompt.

        Args:
            user_prompt: Current user prompt
            include_history: Include conversation history

        Returns:
            Complete prompt string
        """
        parts = []

        # System prompt (always first)
        if self._system_prompt:
            parts.append(self._system_prompt)

        # History context
        if include_history:
            context = self.get_context_for_prompt()
            if context:
                parts.append("\n=== Context ===")
                parts.append(context)

        # Current prompt
        parts.append("\n=== Current Task ===")
        parts.append(user_prompt)

        return "\n".join(parts)

    def prune_history(self, strategy: Optional[str] = None):
        """
        Prune message history based on strategy.

        Args:
            strategy: Pruning strategy (oldest, least_relevant, summarize)
                     If None, uses config default
        """
        strategy = strategy or self.config.prune_strategy

        if strategy == "oldest":
            self._prune_oldest()
        elif strategy == "least_relevant":
            self._prune_least_relevant()
        elif strategy == "summarize":
            self._summarize_history()
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")

    def _prune_oldest(self):
        """Remove oldest messages (keeping system prompt)."""
        # deque automatically handles this with maxlen
        pass

    def _prune_least_relevant(self):
        """Remove least relevant messages based on heuristics."""
        if len(self._messages) <= 3:
            return  # Keep minimum messages

        # Score messages (simple heuristic: length and recency)
        scored = []
        for i, msg in enumerate(self._messages):
            if msg.role == MessageRole.SYSTEM:
                continue  # Never prune system prompt

            # Score: recency (higher for recent) + length
            recency_score = i / len(self._messages)
            length_score = min(len(msg.content) / 1000, 1.0)
            score = recency_score * 0.7 + length_score * 0.3
            scored.append((score, i, msg))

        # Sort by score and keep top messages
        scored.sort(reverse=True)
        keep_count = min(self.config.max_history_size // 2, len(scored))
        keep_indices = {i for _, i, _ in scored[:keep_count]}

        # Rebuild messages
        new_messages = deque(maxlen=self.config.max_history_size)
        for i, msg in enumerate(self._messages):
            if msg.role == MessageRole.SYSTEM or i in keep_indices:
                new_messages.append(msg)

        self._messages = new_messages

    def _summarize_history(self):
        """Summarize old messages (placeholder for future implementation)."""
        # This would use an LLM to summarize old messages
        # For now, just prune oldest
        self._prune_oldest()

    def estimate_token_count(self) -> int:
        """
        Estimate total token count (rough approximation).

        Returns:
            Estimated token count
        """
        total_chars = sum(len(msg.content) for msg in self._messages)
        # Rough estimate: 1 token ~= 4 characters
        return total_chars // 4

    def clear(self):
        """Clear all messages and observations."""
        self._messages.clear()
        self._observations.clear()
        if self._system_prompt:
            self.set_system_prompt(self._system_prompt)

    def clear_observations(self):
        """Clear observations only."""
        self._observations.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "message_count": len(self._messages),
            "observation_count": len(self._observations),
            "estimated_tokens": self.estimate_token_count(),
            "max_history_size": self.config.max_history_size
        }

    def __len__(self) -> int:
        """Get number of messages in history."""
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"MemoryManager("
            f"messages={len(self._messages)}, "
            f"observations={len(self._observations)}, "
            f"tokens~{self.estimate_token_count()})"
        )
