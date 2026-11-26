# Day 4 comparison: LangGraph and Google ADK

| Dimension | LangGraph implementation | Google ADK implementation |
| --- | --- | --- |
| Control flow | Explicit graph nodes, edges, and conditional routing; direct state-machine control. | `SequentialAgent`, `ParallelAgent`, `LoopAgent`, and LLM-agent transfers; concise workflow composition. |
| State | Typed graph state and configurable checkpointers make persisted transitions explicit. | Session state and `output_key` make agent outputs easy to share; persistence depends on the selected session service. |
| Testing | Nodes are ordinary Python functions and are straightforward to test with fake models. | Callback and tool functions are unit-testable; full trajectory evaluation uses ADK evaluation fixtures. |
| Debugging | Graph state, checkpoints, and node traces show the transition path. | ADK events show model calls, tool calls, and state updates in the development UI. |
| Code size | 162 lines (`day3_langgraph/pattern4_supervisor.py`); explicit state, subgraphs, and routing edges. | 136 lines (`day4_adk/pattern6_supervisor.py`); concise `LlmAgent` sub-agent hierarchy with `output_key`. |

Both approaches need explicit tool authorization and offline tests. Use LangGraph
when durable graph state and fine-grained transitions dominate; use ADK when its
agent/session model and workflow primitives fit the service. Five-task runtime
benchmarks remain to be recorded after live provider execution.
