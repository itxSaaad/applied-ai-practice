import logging
import os
import sys
from pathlib import Path
from typing import List, TypedDict
from typing_extensions import Annotated
from dotenv import load_dotenv

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HITL")

SENT_EMAILS_FILE = Path(__file__).resolve().parent.parent / "sent_emails.txt"
CHECKPOINTS_FILE = Path(__file__).resolve().parent.parent / "checkpoints.sqlite"


class EmailState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    draft_email: str
    recipient: str
    status: str


def draft_email_node(state: EmailState) -> dict:
    """Drafts email content payload."""
    draft = "Subject: Incident Postmortem Summary\n\nAll services stabilized."
    recipient = "oncall-team@company.com"
    return {"draft_email": draft, "recipient": recipient, "status": "DRAFTED"}


def send_email_tool_node(state: EmailState) -> dict:
    """Executes human approval pause prior to executing email action side-effect."""
    draft = state["draft_email"]
    recipient = state["recipient"]

    human_decision = interrupt({
        "action": "send_email",
        "recipient": recipient,
        "content": draft,
    })

    decision_type = human_decision.get("type") if isinstance(human_decision, dict) else "reject"

    if decision_type == "approve":
        final_content = draft
    elif decision_type == "edit":
        final_content = human_decision.get("edited_content", draft)
    else:
        return {"status": "REJECTED"}

    try:
        with open(SENT_EMAILS_FILE, "a", encoding="utf-8") as f:
            f.write(f"TO: {recipient}\nCONTENT:\n{final_content}\n---\n")
    except IOError as e:
        return {"status": f"FAILED_SIDE_EFFECT: {str(e)}"}

    return {"status": "SENT", "draft_email": final_content}


def build_hitl_graph(checkpointer):
    builder = StateGraph(EmailState)
    builder.add_node("draft_node", draft_email_node)
    builder.add_node("send_node", send_email_tool_node)

    builder.add_edge(START, "draft_node")
    builder.add_edge("draft_node", "send_node")
    builder.add_edge("send_node", END)
    return builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    callbacks = config.get_callbacks()
    base_config = {"configurable": {"thread_id": "email_thread_1"}}
    if callbacks:
        base_config["callbacks"] = callbacks

    with SqliteSaver.from_conn_string(str(CHECKPOINTS_FILE)) as memory:
        hitl_graph = build_hitl_graph(memory)

        # 1. Initial run: pauses cleanly at interrupt()
        paused = hitl_graph.invoke(
            {"messages": [HumanMessage(content="Send postmortem email")], "draft_email": "", "recipient": "", "status": "INIT"},
            config=base_config,
        )
        print("Paused State Status:", paused.get("status"))
        print("Interrupt Payload:", paused.get("__interrupt__"))

        # 2. Resume after human review/edit
        resumed = hitl_graph.invoke(
            Command(resume={"type": "edit", "edited_content": "Subject: Postmortem\n\nAll services stabilized and verified."}),
            config=base_config,
        )
        config.flush_traces()
        print("Resumed Final Status:", resumed.get("status"))
        print("Draft Content Sent:", resumed.get("draft_email"))
