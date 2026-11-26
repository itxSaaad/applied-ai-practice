from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from day3_langgraph import pattern1_crag as crag
from day3_langgraph import pattern2_hitl as hitl
from day3_langgraph import pattern3_map_reduce as map_reduce
from day3_langgraph import pattern4_supervisor as supervisor


class FakeStructuredModel:
    def __init__(self, value):
        self.value = value

    def invoke(self, _messages, config=None):
        return self.value


class FakeModel:
    def __init__(self, structured_value=None, response="offline answer"):
        self.structured_value = structured_value
        self.response = response

    def with_structured_output(self, _schema):
        return FakeStructuredModel(self.structured_value)

    def invoke(self, _messages, config=None):
        return AIMessage(content=self.response)


def test_crag_routes_retrieves_grades_and_caps_rewrites(monkeypatch):
    question = "Lookup the runbook for checkout timeout"
    monkeypatch.setattr(
        crag,
        "retrieve_runbook_documents",
        lambda _query, k=4: [
            Document(
                page_content="Check upstream latency and request traces.",
                metadata={
                    "source_url": "data/runbooks/checkout_504.md",
                    "section": "Triage",
                },
            )
        ],
    )
    route = crag.route_node({"messages": [HumanMessage(content=question)]})
    assert route.goto == "retrieve_node"
    state = {"question": question}
    documents = crag.retrieve_node(state)["documents"]
    assert documents
    assert any("checkout_504.md" in document for document in documents)

    monkeypatch.setattr(crag, "llm", FakeModel(structured_value=SimpleNamespace(relevant=True)))
    grade = crag.grade_node({"question": question, "documents": documents, "rewrite_count": 0})
    assert grade.goto == "generate_node"

    monkeypatch.setattr(crag, "llm", FakeModel(structured_value=SimpleNamespace(relevant=False)))
    capped = crag.grade_node({"question": question, "documents": ["irrelevant"], "rewrite_count": 2})
    assert capped.goto == "generate_node"
    assert capped.update["documents"] == ["No relevant context found after rewrites."]


def test_crag_rewrite_generate_grounding_and_direct_answer_are_fakeable(monkeypatch):
    fake = FakeModel(structured_value=SimpleNamespace(grounded=False))
    monkeypatch.setattr(crag, "llm", fake)
    rewritten = crag.rewrite_node({"question": "timeouts", "rewrite_count": 0})
    assert rewritten["question"] == "offline answer"
    generated = crag.generate_node({"question": "timeouts", "documents": ["Source: sample.md\nCheck latency."]})
    assert generated["messages"][0].content == "offline answer"
    grounding = crag.check_grounding_node(
        {
            "messages": [AIMessage(content="A result")],
            "documents": ["Source: sample.md"],
            "grounded": False,
        }
    )
    assert grounding.goto == "generate_node"
    assert (
        crag.direct_answer_node({"messages": [HumanMessage(content="hello")]})["messages"][0].content
        == "offline answer"
    )


def test_human_approval_survives_sqlite_restart_and_writes_once(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.sqlite"
    sent_file = tmp_path / "sent_emails.txt"
    monkeypatch.setattr(hitl, "SENT_EMAILS_FILE", sent_file)
    thread_config = {"configurable": {"thread_id": "approval-test-thread"}}
    initial_state = {
        "messages": [HumanMessage(content="Send postmortem email")],
        "draft_email": "",
        "recipient": "",
        "status": "INIT",
    }

    with hitl.SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = hitl.build_hitl_graph(saver)
        paused = graph.invoke(initial_state, config=thread_config)
        assert paused["__interrupt__"]

    with hitl.SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        graph = hitl.build_hitl_graph(saver)
        completed = graph.invoke(
            Command(
                resume={
                    "type": "edit",
                    "edited_content": "Subject: Recovery\n\nService verified.",
                }
            ),
            config=thread_config,
        )
    assert completed["status"] == "SENT"
    contents = sent_file.read_text(encoding="utf-8")
    assert contents.count("SEND_ID: approval-test-thread") == 1
    assert "Service verified." in contents


def test_map_reduce_fans_out_one_work_item_per_document():
    docs = [f"incident signal {index}" for index in range(20)]
    tasks = map_reduce.dynamic_map_fanout({"documents": docs})
    assert len(tasks) == 20


def test_supervisor_can_finish_without_an_external_model(monkeypatch):
    monkeypatch.setattr(
        supervisor,
        "llm",
        FakeModel(structured_value=SimpleNamespace(next_step="FINISH")),
    )
    decision = supervisor.supervisor_node({"messages": [HumanMessage(content="Complete this task")]})
    assert decision.goto == "__end__"
