# Day 2 — LangChain

Day 2 composes model and tool steps with LCEL, validates structured ticket
extraction with Pydantic, and adds bounded tool use and retrieval-augmented
answers.

## Main entry points

| Module | Purpose |
| --- | --- |
| `day2_langchain/pattern1_fallbacks.py` | Primary/fallback model chain. |
| `day2_langchain/pattern2_safe_tool.py` | Read-only incident database tool using `utils.sqlite`. |
| `day2_langchain/pattern3_agent_middleware.py` | Support agent with a bounded model-call budget. |
| `day2_langchain/pattern4_extract_api.py` | FastAPI endpoint and Pydantic ticket schema. |
| `day2_langchain/pattern5_rag_agent.py` | Runbook ingestion, persistent retrieval, and RAG agent. |
| `day2_langchain/chunking_study.py` | Chunking and retrieval experiment. |
| `day2_langchain/batch_extract.py` | Batch ticket extraction and benchmark output. |

`data/sample_tickets.csv`, `data/employee_directory.json`, and
`data/docs_qa.csv` provide tracked examples. The runbooks are under
`data/runbooks/`; persistent embeddings are generated in `artifacts/chroma_db/`.

## Run

```shell
uv run python -m day2_langchain.pattern1_fallbacks
uv run python -m day2_langchain.pattern2_safe_tool
uv run python -m day2_langchain.pattern3_agent_middleware
uv run python -m day2_langchain.pattern5_rag_agent
uv run python -m day2_langchain.chunking_study
uv run python -m day2_langchain.batch_extract
uv run python -m uvicorn day2_langchain.pattern4_extract_api:app --host 127.0.0.1 --port 8000
```

The extraction API uses Gemini. Fallbacks need both Gemini and Ollama;
chunking and RAG need Ollama embeddings. The SQLite tool creates its local
sample database when needed and remains read-only.

## Checks

`evals/test_day2_patterns.py` checks the offline tool, agent, and data
contracts. Versioned extraction examples are in
`evals/datasets/day2_ticket_extraction.json`.
