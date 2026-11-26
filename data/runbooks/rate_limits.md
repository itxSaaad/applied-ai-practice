# Rate-limit responses

Confirm the service quota, actual request rate, shared credentials, and retry policy. Honor `Retry-After` when supplied and use bounded exponential backoff with jitter. Avoid immediate synchronized retries. Verify success rate and request volume return to expected levels before closing the incident.
