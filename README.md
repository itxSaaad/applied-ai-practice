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
├── pyproject.toml        # Pinned project dependencies managed via uv
├── day1_foundations/     # Milestone 1: LLM & ML Foundations for Engineers
├── day2_langchain/       # Milestone 2: LLM Application Engineering with LangChain v1
├── day3_langgraph/       # Milestone 3: Stateful Agents with LangGraph
├── day4_adk/             # Milestone 4: Agents with Google ADK, MCP & A2A
├── day5_observability/   # Milestone 5: Observability, Evaluation & Production Readiness
├── capstone/             # Week 2: OpsPilot Multi-Agent Incident Response Copilot
├── data/                 # Shared data assets, vector stores, and sample documents
├── evals/                # Test datasets and evaluation benchmarks
└── notes/                # Daily notes, decision memos, and architecture records
```

---

## ⚙️ Central Configuration (`config.py`)

All scripts across `day1_foundations/` through `capstone/` load their settings via the root `config.py`:
* **Environment Loading**: Automatically locates and loads the root `.env` regardless of where scripts are executed.
* **LangSmith Auto-Tracing**: Automatically maps `LANGSMITH_TRACING` and `LANGSMITH_API_KEY` to LangChain core standards (`LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY`) without modifying your `.env` file.
* **Langfuse Dual-Tracing**: Instantiates and exports a singleton `CallbackHandler` for LangChain invocations via `config.get_callbacks()`.
* **Operation Spans & Metrics**: `config.trace_operation()` sends non-LangChain work to both providers and logs operation duration and status.
* **Trace Flushing**: `config.flush_traces()` flushes configured LangSmith and Langfuse clients and surfaces delivery errors.

---

## 🚀 Getting Started

### 1. Environment Setup

Ensure Python 3.12 is active and dependencies are installed using `uv`:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install course and evaluation dependencies
uv sync --extra dev
```

### 2. Configure Credentials

Copy `.env.example` to `.env` and fill in your keys:

```ini
# Models
GOOGLE_API_KEY=your_google_ai_studio_key
GOOGLE_GENAI_USE_VERTEXAI=FALSE
LLM_MODEL=gemini-3.5-flash-lite
LOCAL_MODEL=qwen3:8b
USE_LOCAL_LLM=false
OLLAMA_BASE_URL=http://localhost:11434

# LangSmith Tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=agentic-course

# Langfuse Observability
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

---

## 📚 Course Milestones & Deliverables

### Week 1: Learning Milestones

| Milestone | Directory | Focus & Deliverables |
| :--- | :--- | :--- |
| **Day 1: Foundations** | `day1_foundations/` | • Provider wrapper (`llm.py`) switching between Gemini Cloud & local Ollama.<br>• Token & latency profiler (`profile.py`) outputting p50/p95 to CSV.<br>• Brute-force NumPy vs. Chroma vector search (`benchmark_search.py`).<br>• Temperature sampling validity benchmark (`benchmark_temp.py`). |
| **Day 2: LangChain v1** | `day2_langchain/` | • Documentation Q&A RAG tool with Chroma indexing.<br>• FastAPI support ticket structured extraction endpoint with retries.<br>• Chunking study measuring Hit Rate@4 over 25 Q&A pairs.<br>• 3-tool agent with model-call limit middleware.<br>• Fallback test handling Gemini failure with local Ollama. |
| **Day 3: LangGraph** | `day3_langgraph/` | • Corrective-RAG graph (route $\rightarrow$ retrieve $\rightarrow$ grade $\rightarrow$ rewrite $\rightarrow$ generate).<br>• Accuracy & latency comparison between plain and Corrective RAG.<br>• Human approval gate graph with pause, restart, and resume.<br>• Map-reduce summarization flow using `Send`.<br>• Multi-agent supervisor team routing between subgraphs. |
| **Day 4: Google ADK & MCP** | `day4_adk/` | • Incident analyst ADK agent over SQLite incidents and service logs.<br>• Supervisor team ported to Google ADK + LangGraph vs. ADK comparison memo.<br>• Critique loop with `LoopAgent` (writer + critic).<br>• Guardrail callbacks blocking unsafe SQL and redacting PII.<br>• Filesystem MCP tool integration scoping read-only runbook access.<br>• 10-case ADK evaluation set passing $\ge$ 8/10 cases. |
| **Day 5: Observability & Evals** | `day5_observability/` | • Dual tracing setup emitting OpenTelemetry spans to LangSmith & Langfuse.<br>• LangSmith prompt variation experiment on correctness and latency.<br>• Prompt-injection drill measuring resistance before and after defense.<br>• GitHub Actions CI workflow running evaluation tests on pull requests.<br>• 1-page Tool Choice Memo comparing LangSmith and Langfuse. |

### Week 2: Capstone Project (OpsPilot)

* **Directory**: `capstone/`
* **Goal**: Build OpsPilot—an autonomous multi-agent incident response copilot that investigates logs, executes corrective RAG over runbooks, drafts postmortems, and enforces human-in-the-loop approval before taking side-effect actions.

---

## 🧪 Running Day 1 Practice

All Day 1 scripts can be executed directly from either the repo root or inside `day1_foundations/`:

```powershell
# 1. Dual provider check & tracing validation
python day1_foundations/llm.py

# 2. Latency & token profiling (Gemini Cloud or Local Ollama)
python day1_foundations/profile.py --n 5
python day1_foundations/profile.py --local --n 5

# 3. Temperature sampling experiment
python day1_foundations/benchmark_temp.py

# 4. NumPy vs. Chroma vector search benchmark
python day1_foundations/benchmark_search.py
```

## 🧪 Running the Day 4 ADK Evaluation

```powershell
pytest evals/test_day4_adk_evals.py -k "not trajectory" -v
pytest evals/test_day4_adk_evals.py -v
adk eval day4_adk evals/day4_incident_pipeline.evalset.json
```

The ADK trajectory test uses the model selected by `USE_LOCAL_LLM` and can take
several minutes per evaluation run when using a local model.
