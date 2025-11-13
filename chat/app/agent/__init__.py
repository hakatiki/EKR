"""Procurement investigation agent without LangChain/LangGraph."""

from .llm_config import AgentExecutionError, get_agent

__all__ = ["get_agent", "AgentExecutionError"]

