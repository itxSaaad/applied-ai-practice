"""30-case OpsPilot evaluation runner enforcing minimum pass rate for CI."""

from __future__ import annotations

import argparse
import json
from time import perf_counter
from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver

import config
from capstone.data_agent import run_read_only_sql
from capstone.guardrails import detect_prompt_injection, redact_pii, wrap_untrusted_text
from capstone.observability import summarize_system_metrics
from capstone.ops_pilot_supervisor import build_opspilot_graph
from capstone.seed import seed_opspilot_database

if TYPE_CHECKING:
    from pathlib import Path


@config.trace_operation("opspilot_capstone_eval", flush=False)
def run_capstone_eval(
    dataset_path: Path | None = None,
    min_pass_rate: float = 0.8,
) -> dict[str, object]:
    """Run all 30 OpsPilot evaluation cases and return score summary."""
    seed_opspilot_database()
    ds_path = dataset_path or (config.ROOT_DIR / "evals" / "capstone_eval.json")
    payload = json.loads(ds_path.read_text(encoding="utf-8"))
    cases = payload["cases"]

    graph = build_opspilot_graph(checkpointer=MemorySaver(), adk_url="http://127.0.0.1:59999")
    passed = 0
    latencies_ms: list[float] = []
    results: list[dict[str, object]] = []

    for idx, case in enumerate(cases):
        started = perf_counter()
        cat = case["category"]
        q = case["question"]
        ok = False

        if cat in {"knowledge", "data", "adk_fallback"}:
            res = graph.invoke(
                {"question": q},
                config={"configurable": {"thread_id": f"eval-{case['id']}-{idx}"}},
            )
            answer = str(res.get("final_answer", ""))
            if "expected_source" in case:
                ok = case["expected_source"] in answer or case["expected_source"] in res.get("citations", [])
            elif "expected_text" in case:
                ok = case["expected_text"].lower() in answer.lower()
        elif cat == "writer_hitl":
            cfg = {"configurable": {"thread_id": f"eval-hitl-{case['id']}-{idx}"}}
            graph.invoke({"question": q}, config=cfg)
            snap = graph.get_state(cfg)
            ok = bool(snap.next) == bool(case.get("expected_interrupt"))
        elif cat == "security":
            if case.get("expected_blocked_sql"):
                try:
                    run_read_only_sql(q)
                    ok = False
                except ValueError:
                    ok = True
            elif case.get("expected_injection"):
                flags = detect_prompt_injection(q)
                wrapped = wrap_untrusted_text("eval.md", q)
                ok = bool(flags) and "INJECTION_SIGNALS_DETECTED=true" in wrapped
            elif case.get("expected_pii_redacted"):
                masked = redact_pii(q)
                ok = "alice@example.com" not in masked and "555-123-4567" not in masked

        latencies_ms.append((perf_counter() - started) * 1000)
        if ok:
            passed += 1
        results.append({"id": case["id"], "category": cat, "passed": ok})

    pass_rate = passed / len(cases)
    config.log_metric("opspilot_eval_pass_rate", pass_rate, total=len(cases), passed=passed)
    report = {
        "total": len(cases),
        "passed": passed,
        "pass_rate": round(pass_rate, 4),
        "min_pass_rate": min_pass_rate,
        "system_metrics": summarize_system_metrics(latencies_ms),
        "results": results,
    }
    out_file = config.ARTIFACTS_DIR / "evals" / "capstone_eval_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if pass_rate < min_pass_rate:
        raise AssertionError(f"OpsPilot eval pass rate {pass_rate:.1%} < {min_pass_rate:.1%}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 30-case OpsPilot evaluation suite.")
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    args = parser.parse_args()
    summary = run_capstone_eval(min_pass_rate=args.min_pass_rate)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
