# Worker queue backlog

Track queue depth, oldest-message age, worker utilization, completion throughput, and dependency failures. Confirm healthy workers and downstream capacity before changing concurrency. A rising queue can be caused by slow jobs or retries rather than too few workers. Verify that backlog age and error rate return to baseline after mitigation.
