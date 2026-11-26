# Authentication token refresh and cache eviction

## Triage
1. Check authentication error rate, refresh latency, cache hit rate, and
   connection-pool waits for the affected time window.
2. Compare token expiry and refresh behavior across a small, approved sample;
   never copy bearer tokens or credentials into tickets or logs.
3. Check cache eviction, provider health, and recent configuration or deploy
   changes.

## Mitigation
1. Restore the last known safe cache configuration if eviction behavior changed.
2. Protect the identity provider with bounded retries, jitter, and its
   `Retry-After` guidance; avoid broad cache flushes.
3. Escalate suspected credential compromise through the approved security
   process rather than exposing tokens during diagnosis.

## Recovery
1. Verify refresh success, authentication latency, cache hit rate, and a
   representative sign-in flow.
2. Confirm no secrets were retained in logs and record the recovery evidence
   and owner for follow-up work.
