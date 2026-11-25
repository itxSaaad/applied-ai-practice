"""Day 4: Google ADK 2.x, MCP Toolsets, and A2A Agent-to-Agent Architecture."""

__all__ = ["agent", "root_agent"]


def __getattr__(name: str):
    if name == "agent":
        from importlib import import_module

        return import_module(".agent", __name__)
    if name == "root_agent":
        from .agent import root_agent

        return root_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
