# Tool choice memo: LangSmith vs Langfuse

## Recommendation

Use both during this course: LangSmith for the strongest LangChain/LangGraph
tracing and dataset experiments; Langfuse for framework-neutral tracing, prompt
labels, and the Google ADK OpenTelemetry path. Keep application code dependent
on the local callback/configuration helpers so either backend can be disabled.

## Decision factors

| Factor | LangSmith | Langfuse |
| --- | --- | --- |
| Hosting and residency | Managed service; verify the selected region and retention terms. | Cloud or self-hosted; self-hosting gives more control over residency but adds operations. |
| Framework coverage | Deep LangChain/LangGraph integration; other frameworks use OpenTelemetry. | Framework-agnostic OpenTelemetry plus integrations including LangChain and ADK. |
| Cost | Depends on current plan and monthly trace volume; sample before production. | Cloud plan limits and self-hosting infrastructure costs change; measure units and retention needs. |
| Team fit | Best when agent debugging and LangChain experiments are primary. | Best when prompt management, cross-framework tracing, or self-hosting are primary. |

No volume, residency region, or plan has been supplied for the team, so a numeric
production cost or residency claim would be misleading. Before choosing a
production default, record expected runs/day, average spans/run, retention, data
classification, and required region; compare those inputs against each provider's
current terms and price calculator.
