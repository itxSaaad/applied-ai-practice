# Database connection pool saturation

Compare active and waiting connections with configured pool limits. Inspect connection wait time, transaction duration, slow queries, and recent workload changes. Check for leaked or long-lived transactions before increasing pool size; raising the limit may move saturation to the database. Verify recovery through bounded query latency and normal connection waits.
