"""
Defect Prediction Agent - Robust core architecture.

This module provides a framework-free agent implementation for software defect prediction
using Large Language Models and external tools.
"""

from .config import SystemConfig, LLMConfig, AgentConfig, MemoryConfig
from .core import DefectPredictionOrchestrator, AgentContext, DefectPrediction
from .tools import ToolRegistry
from .tools.implementations import WebSearchTool, DocumentationSearchTool, GitRepositoryTool, CommitData
from .commit import (
    RepositoryAnalysisPipeline,
    CommitDefectOrchestrator,
    CommitDefectResult,
    CommitIntentAnalyzer,
    CommitIntent
)

__version__ = "2.0.0"

__all__ = [
    "SystemConfig",
    "LLMConfig",
    "AgentConfig",
    "MemoryConfig",
    "DefectPredictionOrchestrator",
    "AgentContext",
    "DefectPrediction",
    "ToolRegistry",
    "WebSearchTool",
    "DocumentationSearchTool",
    "GitRepositoryTool",
    "CommitData",
    "RepositoryAnalysisPipeline",
    "CommitDefectOrchestrator",
    "CommitDefectResult",
    "CommitIntentAnalyzer",
    "CommitIntent"
]


# Convenience function for quick setup
def create_agent(
    model_name: str = "codegemma:7b",
    enable_web_search: bool = True,
    enable_doc_search: bool = True,
    verbose: bool = True
):
    """
    Create a defect prediction agent with default configuration.

    Args:
        model_name: LLM model to use
        enable_web_search: Enable web search tool
        enable_doc_search: Enable documentation search tool
        verbose: Print execution details

    Returns:
        Configured DefectPredictionOrchestrator
    """
    # Create configuration
    config = SystemConfig.default()
    config.llm.model_name = model_name
    config.agent.enable_web_search = enable_web_search
    config.agent.enable_doc_search = enable_doc_search
    config.agent.verbose = verbose

    # Create orchestrator
    agent = DefectPredictionOrchestrator(config)

    # Register tools
    if enable_web_search:
        agent.register_tool(WebSearchTool())

    if enable_doc_search:
        agent.register_tool(DocumentationSearchTool())

    return agent


def create_repository_analyzer(
    model_name: str = "codegemma:7b",
    enable_web_search: bool = True,
    enable_doc_search: bool = True,
    verbose: bool = True
):
    """
    Create a repository analysis pipeline.

    Args:
        model_name: LLM model to use
        enable_web_search: Enable web search for verifying package APIs and behaviors
        enable_doc_search: Enable documentation search for checking library updates
        verbose: Print execution details

    Returns:
        Configured RepositoryAnalysisPipeline
    """
    # Create configuration
    config = SystemConfig.default()
    config.llm.model_name = model_name
    config.agent.enable_web_search = enable_web_search
    config.agent.enable_doc_search = enable_doc_search
    config.agent.verbose = verbose

    # Create pipeline
    pipeline = RepositoryAnalysisPipeline(config, verbose=verbose)

    # Register tools if enabled
    if enable_web_search:
        pipeline.register_tool(WebSearchTool())

    if enable_doc_search:
        pipeline.register_tool(DocumentationSearchTool())

    return pipeline
