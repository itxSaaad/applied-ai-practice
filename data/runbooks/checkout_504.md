# Checkout upstream timeout (HTTP 504)

## Symptoms

Checkout requests exceed the gateway timeout while calling a downstream dependency. Confirm the affected route, error rate, latency percentiles, and request IDs.

## Response

1. Compare checkout latency with payment and authentication dependency latency.
2. Inspect traces and recent deploys for the affected time window.
3. Check retry volume and queue depth; retries can amplify an upstream outage.
4. Mitigate only through a reviewed change or a safe rollback.
5. Verify with a test purchase and sustained normal latency and error rate.

Do not increase timeouts or restart dependencies without checking their health and capacity.
