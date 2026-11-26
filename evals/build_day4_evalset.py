"""Regenerate the ten-case ADK evaluation set from its concise source cases."""

import json

import config

OUTPUT = config.ROOT_DIR / "evals" / "datasets" / "day4_incident_pipeline.evalset.json"
CASES = (
    (
        "case_01_auth_sweep",
        "Sweep auth-service logs for cache and token errors.",
        "Auth cache hit rate fell to 42 percent; token refresh latency peaked, then improved after the cache warmed.",
    ),
    (
        "case_02_payment_sweep",
        "Review payment-service logs for upstream latency.",
        "Payment provider latency peaked and then recovered; verify the current provider health before closing.",
    ),
    (
        "case_03_checkout_sweep",
        "Check checkout-service logs for HTTP 504 errors.",
        "Checkout returned 504s while payment requests timed out; retries were capped and the error rate trended down.",
    ),
    (
        "case_04_worker_sweep",
        "Review the worker-service queue health from the sample logs.",
        "Worker queue age reached nine minutes before throughput returned to baseline.",
    ),
    (
        "case_05_active_critical",
        "Find active critical incidents in the incidents database.",
        "INC001 is a critical checkout-api incident and is currently investigating.",
    ),
    (
        "case_06_checkout_incidents",
        "List incidents affecting checkout-service.",
        "The incident records include checkout-service gateway timeouts and retry-budget issues.",
    ),
    (
        "case_07_auth_incidents",
        "Find incidents for auth-service.",
        "The incident records include authentication latency and token-refresh issues.",
    ),
    (
        "case_08_incident_status",
        "Summarize the current status of the incident records.",
        "The incident table contains records across investigating, identified, monitoring, resolved, and closed states.",
    ),
    (
        "case_09_payment_incidents",
        "Find payment-service incidents and inspect payment logs.",
        "Payment-service records include provider latency and connection-reset incidents; the sample log shows provider latency recovering.",
    ),
    (
        "case_10_incident_sweep",
        "Run a complete incident triage and log sweep.",
        "The sample evidence links auth cache degradation with checkout timeouts; payment latency and worker throughput recovered.",
    ),
)

EXPECTED_TOOLS = [
    {"name": "query_incidents"},
    {"name": "read_log", "args": {"service": "auth-service", "minutes": 15}},
    {"name": "read_log", "args": {"service": "payment-service", "minutes": 15}},
    {"name": "read_log", "args": {"service": "checkout-service", "minutes": 15}},
    {"name": "read_log", "args": {"service": "worker-service", "minutes": 15}},
]


def build_case(eval_id: str, prompt: str, expected_response: str) -> dict:
    return {
        "eval_id": eval_id,
        "conversation": [
            {
                "invocation_id": f"inv_{eval_id}",
                "user_content": {"parts": [{"text": prompt}], "role": "user"},
                "final_response": {
                    "parts": [{"text": expected_response}],
                    "role": "model",
                },
                "intermediate_data": {"tool_uses": [dict(tool) for tool in EXPECTED_TOOLS]},
            }
        ],
    }


if __name__ == "__main__":
    dataset = {
        "eval_set_id": "day4_incident_pipeline_trajectory",
        "name": "Day 4 Incident Triage and Log Sweep",
        "description": (
            "Ten deterministic prompts exercising read-only database triage and "
            "parallel auth, payment, and checkout log tools."
        ),
        "eval_cases": [build_case(eval_id, prompt, expected_response) for eval_id, prompt, expected_response in CASES],
    }
    OUTPUT.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(CASES)} evaluation cases to {OUTPUT}")
