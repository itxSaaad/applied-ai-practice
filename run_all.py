"""Run the offline acceptance checks for Course 1 Days 1–5."""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import config

SUITES = {
    "day1": ("evals/test_day1_foundations.py",),
    "day2": ("evals/test_day2_patterns.py",),
    "day3": ("day3_langgraph/test_graph.py",),
    "day4": (
        "evals/test_day4_guardrails_offline.py",
        "evals/test_day4_adk_evals.py",
        "evals/test_trajectories.py",
    ),
    "day5": (
        "evals/test_day5_observability.py",
        "evals/test_unit_guardrails.py",
        "evals/test_rate_limit.py",
        "evals/test_ci_gate.py",
    ),
}


def _append_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as ledger:
        ledger.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_suite(day: str, relative_file: str, run_id: str, log_dir: Path, ledger: Path) -> bool:
    source = config.ROOT_DIR / relative_file
    log_path = log_dir / f"{day}_{source.stem}.log"
    env = config.get_offline_test_environment(
        course_day=day,
        source_file=relative_file,
        run_id=run_id,
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-m",
        "not integration",
        str(source),
    ]
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=config.ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    output = result.stdout + result.stderr
    log_path.write_text(output, encoding="utf-8")
    status = "passed" if result.returncode == 0 else "failed"
    event = {
        "timestamp": started_at.isoformat(),
        "day": day,
        "source_file": relative_file.replace("\\", "/"),
        "run_id": run_id,
        "status": status,
        "duration_ms": duration_ms,
        "exit_code": result.returncode,
        "log_file": log_path.relative_to(config.ROOT_DIR).as_posix(),
        "command": command,
    }
    _append_event(ledger, event)
    print(f"{day} {relative_file}: {status} ({duration_ms:.0f} ms)")
    if output.strip():
        console_encoding = sys.stdout.encoding or "utf-8"
        safe_output = output.encode(console_encoding, errors="replace").decode(console_encoding, errors="replace")
        print(safe_output.rstrip())
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline Course 1 acceptance tests for Days 1–5.")
    parser.add_argument(
        "--day",
        choices=tuple(SUITES),
        action="append",
        help="Run selected day(s); repeat this option to select several.",
    )
    args = parser.parse_args()

    run_id = uuid4().hex
    day_names = args.day or list(SUITES)
    date_folder = datetime.now().astimezone().date().isoformat()
    log_dir = config.LOGS_DIR / date_folder
    log_dir.mkdir(parents=True, exist_ok=True)
    ledger = log_dir / "runs.jsonl"

    outcomes = [
        run_suite(day, relative_file, run_id, log_dir, ledger) for day in day_names for relative_file in SUITES[day]
    ]
    passed = sum(outcomes)
    print(
        f"Course 1 offline checks: {passed}/{len(outcomes)} passed. "
        f"Run {run_id}; ledger: {ledger.relative_to(config.ROOT_DIR).as_posix()}"
    )
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
