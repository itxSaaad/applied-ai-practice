from pathlib import Path
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

import config

SENT_EMAILS_FILE = Path(config.SENT_EMAILS_PATH)
CHECKPOINTS_FILE = Path(config.CHECKPOINTS_DB_PATH)


class EmailState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    draft_email: str
    recipient: str
    status: str


def draft_email_node(state: EmailState) -> dict:
    """Drafts email content payload."""
    draft = "Subject: Incident Postmortem Summary\n\nAll services stabilized."
    recipient = "oncall-team@company.com"
    return {"draft_email": draft, "recipient": recipient, "status": "DRAFTED"}


def send_email_tool_node(state: EmailState, config: RunnableConfig) -> dict:
    """Executes human approval pause prior to executing email action side-effect."""
    draft = state["draft_email"]
    recipient = state["recipient"]
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        raise ValueError("A thread_id is required to make the email side effect idempotent.")

    human_decision = interrupt(
        {
            "action": "send_email",
            "recipient": recipient,
            "content": draft,
        }
    )

    decision_type = human_decision.get("type") if isinstance(human_decision, dict) else "reject"

    if decision_type == "approve":
        final_content = draft
    elif decision_type == "edit":
        final_content = human_decision.get("edited_content", draft)
    else:
        return {"status": "REJECTED"}

    send_marker = f"SEND_ID: {thread_id}"
    existing = SENT_EMAILS_FILE.read_text(encoding="utf-8") if SENT_EMAILS_FILE.exists() else ""
    if send_marker not in existing:
        with SENT_EMAILS_FILE.open("a", encoding="utf-8") as file:
            file.write(f"{send_marker}\nTO: {recipient}\nCONTENT:\n{final_content}\n---\n")

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
    thread_id = f"email-{uuid4().hex}"
    base_config = {
        "configurable": {"thread_id": thread_id},
        **config.get_run_config(session_id=thread_id),
    }

    with SqliteSaver.from_conn_string(str(CHECKPOINTS_FILE)) as memory:
        hitl_graph = build_hitl_graph(memory)

        # 1. Initial run: pauses cleanly at interrupt()
        paused = hitl_graph.invoke(
            {
                "messages": [HumanMessage(content="Send postmortem email")],
                "draft_email": "",
                "recipient": "",
                "status": "INIT",
            },
            config=base_config,
        )
        print("Paused State Status:", paused.get("status"))
        print("Interrupt Payload:", paused.get("__interrupt__"))

        # 2. Resume after human review/edit
        resumed = hitl_graph.invoke(
            Command(
                resume={
                    "type": "edit",
                    "edited_content": "Subject: Postmortem\n\nAll services stabilized and verified.",
                }
            ),
            config=base_config,
        )
        config.flush_traces()
        print("Resumed Final Status:", resumed.get("status"))
        print("Draft Content Sent:", resumed.get("draft_email"))
