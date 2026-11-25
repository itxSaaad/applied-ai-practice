# Applied AI & Agentic Engineering Practice

Hands-on repository for the **Applied AI & Agentic Engineering** course. This repository tracks daily learning milestones (Week 1) and the end-to-end multi-agent capstone project (Week 2, **OpsPilot**).

---

## 📁 Repository Structure

All exercise scripts and deliverables are organized strictly within their respective milestone directories. The repository root contains only shared tooling, environment definitions, and project configuration:

```text
applied-ai-practice/
├── .env                  # Environment variables & API keys (Never committed to Git)
├── .env.example          # Sanitized template for environment configuration
├── .gitignore            # Git ignore rules for virtualenvs, caches, and secrets
├── config.py             # Central environment loader & dual tracing bridge
├── llm.py                # Shared Gemini/Ollama wrapper
├── pyproject.toml        # Pinned project dependencies managed via uv
├── day1_foundations/     # Day 1 scripts and profiling CSVs
├── day2_langchain/       # Milestone 2: LLM Application Engineering with LangChain v1
├── day3_langgraph/       # Milestone 3: Stateful Agents with LangGraph
├── day4_adk/             # Milestone 4: Agents with Google ADK, MCP & A2A
├── day5_observability/   # Scaffold for Day 5 deliverables
├── capstone/             # Scaffold for the Days 6–10 OpsPilot project
├── data/                 # Shared sample data and runbooks
├── evals/                # Test datasets and evaluation benchmarks
└── notes/                # Daily notes, decision memos, and architecture records
```

---

## ⚙️ Central Configuration (`config.py`)

All scripts across `day1_foundations/` through `capstone/` load their settings via the root `config.py`:
* **Environment Loading**: Automatically locates and loads the root `.env` regardless of where scripts are executed.
* **Ollama URL**: `OLLAMA_BASE_URL` is the only supported environment variable for the Ollama server URL. Ollama integrations read `config.OLLAMA_BASE_URL` (defaults to `http://localhost:11434`).
* **LangSmith Auto-Tracing**: Automatically maps `LANGSMITH_TRACING` and `LANGSMITH_API_KEY` to LangChain core standards (`LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY`) without modifying your `.env` file.
* **Langfuse Dual-Tracing**: Instantiates and exports a singleton `CallbackHandler` for LangChain invocations via `config.get_callbacks()`.
* **Operation Spans & Metrics**: `config.trace_operation()` sends non-LangChain work to both providers and logs operation duration and status.
* **Trace Flushing**: `config.flush_traces()` flushes configured LangSmith and Langfuse clients and surfaces delivery errors.

---

## 🚀 Getting Started

### Prerequisites

- Windows PowerShell and the repository checked out locally.
- The trusted global Python at `C:\Python314\python.exe` and `uv`. Do not create a workspace `.venv` on `D:`.
- Ollama installed and running for local-model exercises. Install Node.js/npm for the Day 4 MCP filesystem exercise.

Install `uv` and Ollama before continuing. Node.js/npm are only needed for the MCP exercise.

### 1. Install Python dependencies

Run these commands from the repository root:

```powershell
C:\Python314\python.exe --version
uv --version
ollama --version
uv pip install --python C:\Python314\python.exe --system --break-system-packages -e ".[dev]"
```

### 2. Create and configure `.env`

Create `.env` from the template only when one does not already exist. This avoids overwriting existing credentials:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

For local Ollama runs, set these values in `.env`:

```ini
USE_LOCAL_LLM=true
LOCAL_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
```

Then start Ollama and download the chat and embedding models used by the exercises:

```powershell
ollama pull qwen3:8b
ollama pull nomic-embed-text
ollama list
```

`GOOGLE_API_KEY` is required for exercises that call Gemini, including the Day 2 extraction API and temperature benchmark. Set `USE_LOCAL_LLM=false` to select Gemini in shared-model and ADK examples. LangSmith/Langfuse credentials are optional; leave their keys blank and tracing disabled for a local-only setup. Never replace an existing `.env` with the template.

### 3. Check configuration

From the repository root, confirm the Python package and configured URL load:

```powershell
C:\Python314\python.exe -c "import config; print(config.OLLAMA_BASE_URL)"
```

---

## 📚 Course Milestones & Deliverables

The suggested curriculum names below are logical descriptions. Existing equivalent files are kept under their current names; files marked **planned** are not yet implemented.

| Milestone | Directory | Focus & Deliverables |
| :--- | :--- | :--- |
| **Day 1: Foundations** | `day1_foundations/` | `profile.py`; `benchmark_search.py` (semantic-search exercise); `benchmark_temp.py` (temperature experiment); tracked profile CSVs. The proposed `decision_memo.md` is planned, not present. |
| **Day 2: LangChain v1** | `day2_langchain/` | `pattern1_fallbacks.py` (fallback test); `pattern2_safe_tool.py`; `pattern3_agent_middleware.py` (tool agent); `pattern4_extract_api.py` (extraction API); `pattern5_rag_agent.py`. A separate chunking-study experiment is not present yet. |
| **Day 3: LangGraph** | `day3_langgraph/` | `pattern1_crag.py`; `pattern2_hitl.py`; `pattern3_map_reduce.py`; `pattern4_supervisor.py`; `pattern5_functional_api.py`. A dedicated offline `test_graph.py` suite is not present yet. |
| **Day 4: Google ADK, MCP & A2A** | `day4_adk/`, `evals/` | `agent.py`, `model_config.py`, `pattern1_pipeline.py`, `pattern2_critique_loop.py`, `pattern3_guardrail_callbacks.py`, `pattern4_mcp_runbooks.py`, `pattern5_a2a_server.py`, `test_a2a_client.py`; `evals/day4_incident_pipeline.evalset.json` is the current eval-set file. The current pipeline is not an ADK supervisor port. |
| **Day 5: Observability & Evaluation** | `day5_observability/` | Directory scaffold only; the listed tracing, prompt experiment, injection drill, CI workflow, and memo are planned. |
| **Days 6–10: OpsPilot Capstone** | `capstone/` | Directory scaffold only; the listed container setup and incident/data agents are planned. |

### Planned deliverables not present yet

The following items have no implementation in the current workspace: Day 1 `decision_memo.md`; Day 2 `chunking_study.py`; Day 3 offline `test_graph.py`; the Day 4 ADK supervisor port/comparison memo; all Day 5 files (`tracing_setup.py`, `eval_experiment.py`, `injection_drill.py`, `.github/workflows/eval_ci.yml`, `tool_choice_memo.md`); and all Days 6–10 capstone files (`docker-compose.yml`, `knowledge_agent.py`, `data_agent.py`, `incident_analyst_a2a.py`, `ops_pilot_supervisor.py`). The existing pattern files listed above are kept under their current names rather than duplicated or renamed.

### Run Day 1

The dual-provider wrapper exercises both Ollama and Gemini, so it requires a running Ollama server and a valid `GOOGLE_API_KEY`. For a local-only profile run, use `--local`.

```powershell
C:\Python314\python.exe llm.py
C:\Python314\python.exe day1_foundations\profile.py --local --n 5
C:\Python314\python.exe day1_foundations\profile.py --n 5
C:\Python314\python.exe day1_foundations\benchmark_search.py
C:\Python314\python.exe day1_foundations\benchmark_temp.py
```

The cloud profile and temperature experiment require Gemini credentials. Semantic search and local profiling require Ollama; the RAG exercise also needs `nomic-embed-text`.

### Run Day 2

Run the standalone patterns from the repository root:

```powershell
C:\Python314\python.exe -m day2_langchain.pattern1_fallbacks
C:\Python314\python.exe -m day2_langchain.pattern2_safe_tool
C:\Python314\python.exe -m day2_langchain.pattern3_agent_middleware
C:\Python314\python.exe -m day2_langchain.pattern5_rag_agent
```

Run the extraction API in a separate terminal; open `http://127.0.0.1:8000/docs` to try its POST endpoint:

```powershell
C:\Python314\python.exe -m uvicorn day2_langchain.pattern4_extract_api:app --host 127.0.0.1 --port 8000
```

The extraction API uses Gemini. The fallback example needs both Gemini credentials and a reachable local Ollama server. The RAG example uses Ollama embeddings and Gemini for answer generation.

### Run Day 3

```powershell
C:\Python314\python.exe -m day3_langgraph.pattern1_crag
C:\Python314\python.exe -m day3_langgraph.pattern2_hitl
C:\Python314\python.exe -m day3_langgraph.pattern3_map_reduce
C:\Python314\python.exe -m day3_langgraph.pattern4_supervisor
C:\Python314\python.exe -m day3_langgraph.pattern5_functional_api
```

### Run Day 4

```powershell
C:\Python314\python.exe -m day4_adk.pattern1_pipeline
C:\Python314\python.exe -m day4_adk.pattern2_critique_loop
C:\Python314\python.exe -m day4_adk.pattern3_guardrail_callbacks
C:\Python314\python.exe -m day4_adk.pattern4_mcp_runbooks
```

The MCP exercise requires Node.js/npm so it can start `@modelcontextprotocol/server-filesystem`. To test A2A, start the server in one terminal and the client in another:

```powershell
C:\Python314\python.exe -m day4_adk.pattern5_a2a_server
C:\Python314\python.exe -m day4_adk.test_a2a_client
```

To run the ADK entrypoint or evaluation:

```powershell
C:\Python314\Scripts\adk.exe run day4_adk
C:\Python314\Scripts\adk.exe web day4_adk
C:\Python314\python.exe -m pytest evals\test_day4_adk_evals.py -v
C:\Python314\Scripts\adk.exe eval day4_adk evals\day4_incident_pipeline.evalset.json
```

The trajectory evaluation makes model calls and may take several minutes with a local model.
