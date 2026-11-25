import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from google.adk.agents import LlmAgent
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioConnectionParams, StdioServerParameters
from day4_adk.model_config import get_adk_model

# Scoped runbooks directory path
RUNBOOK_DIR = str(REPO_ROOT / "data" / "runbooks")

# Configure Stdio connection parameters for Model Context Protocol filesystem server
params = StdioConnectionParams(
    server_params=StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", RUNBOOK_DIR],
    ),
    timeout=60.0,
)

# Instantiate McpToolset bound to local filesystem server
runbook_mcp_toolset = McpToolset(connection_params=params)

# Attach toolset to ADK agent
runbook_agent = LlmAgent(
    name="runbook_agent",
    model=get_adk_model(),
    instruction=(
        "You diagnose system incidents by reading and executing runbooks from the scoped filesystem. "
        "Locate relevant runbook files in data/runbooks to guide troubleshooting."
    ),
    tools=[runbook_mcp_toolset],
)


if __name__ == "__main__":
    print(f"Runbook MCP Toolset initialized for directory: {RUNBOOK_DIR}")
    print(f"Runbook Agent: {runbook_agent.name} configured with MCP tools.")
    config.flush_traces()
