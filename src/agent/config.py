"""
Configuration management for the defect prediction agent.
Centralizes all configuration settings with environment variable support.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    model_name: str = "codegemma:7b"
    api_url: str = "http://localhost:11434/api/generate"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: int = 120
    stream: bool = False

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create configuration from environment variables."""
        return cls(
            model_name=os.getenv("AGENT_MODEL_NAME", cls.model_name),
            api_url=os.getenv("OLLAMA_API_URL", cls.api_url),
            temperature=float(os.getenv("AGENT_TEMPERATURE", cls.temperature)),
            max_tokens=int(os.getenv("AGENT_MAX_TOKENS", cls.max_tokens)),
            timeout=int(os.getenv("AGENT_TIMEOUT", cls.timeout)),
            stream=os.getenv("AGENT_STREAM", "false").lower() == "true"
        )


@dataclass
class AgentConfig:
    """Configuration for agent orchestrator."""
    max_iterations: int = 5
    confidence_threshold: float = 0.7
    enable_web_search: bool = True
    enable_doc_search: bool = True
    enable_pdf_reader: bool = False
    parallel_tools: bool = False
    verbose: bool = True

    # Confidence detection settings
    min_observation_lines: int = 3
    defect_keywords: list = field(default_factory=lambda: [
        r'\berror\b', r'\bbug\b', r'\bvulnerab',
        r'\bfail\b', r'\bexception\b', r'\bnull\b',
        r'\bindexerror\b', r'\btypeerror\b', r'\bcritical\b'
    ])
    clean_keywords: list = field(default_factory=lambda: [
        r'\bno issues?\b', r'\bno defects?\b',
        r'\bclean\b', r'\bpassed\b', r'\bsafe\b'
    ])

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables."""
        return cls(
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", cls.max_iterations)),
            confidence_threshold=float(os.getenv("AGENT_CONFIDENCE_THRESHOLD", cls.confidence_threshold)),
            enable_web_search=os.getenv("AGENT_ENABLE_WEB_SEARCH", "true").lower() == "true",
            enable_doc_search=os.getenv("AGENT_ENABLE_DOC_SEARCH", "true").lower() == "true",
            enable_pdf_reader=os.getenv("AGENT_ENABLE_PDF_READER", "false").lower() == "true",
            parallel_tools=os.getenv("AGENT_PARALLEL_TOOLS", "false").lower() == "true",
            verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true"
        )


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_history_size: int = 20
    max_context_tokens: int = 4096
    enable_summarization: bool = False
    prune_strategy: str = "oldest"  # oldest, least_relevant, summarize

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create configuration from environment variables."""
        return cls(
            max_history_size=int(os.getenv("AGENT_MAX_HISTORY", cls.max_history_size)),
            max_context_tokens=int(os.getenv("AGENT_MAX_CONTEXT_TOKENS", cls.max_context_tokens)),
            enable_summarization=os.getenv("AGENT_ENABLE_SUMMARIZATION", "false").lower() == "true",
            prune_strategy=os.getenv("AGENT_PRUNE_STRATEGY", cls.prune_strategy)
        )


@dataclass
class SystemConfig:
    """Main system configuration aggregating all sub-configurations."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Create full system configuration from environment variables."""
        return cls(
            llm=LLMConfig.from_env(),
            agent=AgentConfig.from_env(),
            memory=MemoryConfig.from_env()
        )

    @classmethod
    def default(cls) -> "SystemConfig":
        """Create default configuration."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "llm": self.llm.__dict__,
            "agent": self.agent.__dict__,
            "memory": self.memory.__dict__
        }

    def update(self, **kwargs) -> "SystemConfig":
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
