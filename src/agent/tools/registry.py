"""
Tool registry for dynamic tool registration and execution.
"""

import time
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import BaseTool, ToolException, ToolMetadata
from ..core.state import ToolCall, ToolResult


class ToolRegistry:
    """
    Central registry for managing and executing tools.

    Provides dynamic tool registration, discovery, and execution
    with support for parallel execution.
    """

    def __init__(self, max_workers: int = 3):
        self._tools: Dict[str, BaseTool] = {}
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If tool with same name already exists
        """
        tool_name = tool.name
        if tool_name in self._tools:
            raise ValueError(f"Tool '{tool_name}' is already registered")
        self._tools[tool_name] = tool

    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.

        Args:
            tool_name: Name of tool to unregister

        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of tool to retrieve

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)

    def has(self, tool_name: str) -> bool:
        """Check if tool exists in registry."""
        return tool_name in self._tools

    def list_tools(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())

    def get_all_metadata(self) -> List[ToolMetadata]:
        """Get metadata for all registered tools."""
        return [tool.get_metadata() for tool in self._tools.values()]

    def execute(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
        """
        Execute a single tool.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        start_time = time.time()

        # Check if tool exists
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found",
                execution_time=0.0
            )

        # Execute tool
        try:
            success, result, error = tool(**kwargs)
            execution_time = time.time() - start_time

            return ToolResult(
                tool_name=tool_name,
                success=success,
                result=result,
                error=error,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Unexpected error: {str(e)}",
                execution_time=execution_time
            )

    def execute_multiple(
        self,
        tool_calls: List[ToolCall],
        parallel: bool = False
    ) -> List[ToolResult]:
        """
        Execute multiple tools sequentially or in parallel.

        Args:
            tool_calls: List of tool calls to execute
            parallel: If True, execute tools in parallel

        Returns:
            List of ToolResult objects
        """
        if not tool_calls:
            return []

        if parallel:
            return self._execute_parallel(tool_calls)
        else:
            return self._execute_sequential(tool_calls)

    def _execute_sequential(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute tools sequentially."""
        results = []
        for tool_call in tool_calls:
            result = self.execute(
                tool_call.tool_name,
                **tool_call.arguments
            )
            results.append(result)
        return results

    def _execute_parallel(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute tools in parallel using thread pool."""
        futures = {}

        # Submit all tasks
        for tool_call in tool_calls:
            future = self._executor.submit(
                self.execute,
                tool_call.tool_name,
                **tool_call.arguments
            )
            futures[future] = tool_call.tool_name

        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Handle unexpected errors in thread execution
                tool_name = futures[future]
                results.append(ToolResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=f"Thread execution error: {str(e)}",
                    execution_time=0.0
                ))

        return results

    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """Get description for a specific tool."""
        tool = self.get(tool_name)
        return tool.description if tool else None

    def format_tools_for_prompt(self) -> str:
        """
        Format all tools as a string suitable for LLM prompts.

        Returns:
            Formatted string describing all available tools
        """
        if not self._tools:
            return "No tools available."

        lines = ["Available tools:"]
        for tool_name, tool in self._tools.items():
            metadata = tool.get_metadata()
            lines.append(f"\n{tool_name}:")
            lines.append(f"  Description: {metadata.description}")
            lines.append(f"  Parameters: {metadata.parameters}")
            lines.append(f"  Returns: {metadata.returns}")
            if metadata.examples:
                lines.append(f"  Examples: {metadata.examples}")

        return "\n".join(lines)

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if tool exists using 'in' operator."""
        return self.has(tool_name)
