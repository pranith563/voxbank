"""
llm_agent.py

Compatibility shim for legacy imports.

The actual agent implementation now lives in orchestrator/src/agent/agent.py
as VoxBankAgent. This module simply re-exports that class as LLMAgent so
existing code that imports from `llm_agent` continues to work.
"""

from agent.agent import VoxBankAgent as LLMAgent  # noqa: F401

