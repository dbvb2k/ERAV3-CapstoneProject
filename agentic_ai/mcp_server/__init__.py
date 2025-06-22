"""
MCP (Model Context Protocol) server initialization.
This module provides the server implementation for tool calling and agent coordination.
"""

from .server import MCPServer
from .tools import register_tools

__all__ = ['MCPServer', 'register_tools'] 