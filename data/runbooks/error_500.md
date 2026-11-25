# HTTP 500 Troubleshooting Runbook

Use this runbook during MCP filesystem tool tests. Keep investigation read-only until an operator approves a change.

1. Confirm the affected service, first failure time, and request or trace identifier.
2. Read the service logs for the last 15 minutes and identify the first HTTP 500 response.
3. Check recent deployments, dependency health, saturation, and error-rate changes around that time.
4. Compare a failing request with a successful request and record the smallest reproducible difference.
5. If a recent deployment matches the start time, ask the incident commander whether to roll it back.
6. Verify recovery with health checks and a representative request, then document the cause and actions.
