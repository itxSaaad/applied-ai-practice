"""Run the structured extraction acceptance set against the configured model."""

import argparse
import csv
from pathlib import Path

import config
from day2_langchain.pattern4_extract_api import Ticket, build_structured_extractor
from utils.logging import get_logger

logger = get_logger(__name__)


def run_batch(
    *,
    input_path: Path = config.DATA_DIR / "sample_tickets.csv",
    output_path: Path = config.BENCHMARKS_DIR / "ticket_extraction_results.csv",
    minimum_valid: int = 19,
) -> dict[str, int | float]:
    with input_path.open(encoding="utf-8", newline="") as file:
        tickets = list(csv.DictReader(file))
    if len(tickets) != 20:
        raise ValueError(f"Expected 20 sample tickets; found {len(tickets)}.")

    extractor = build_structured_extractor()
    results = []
    valid_count = 0
    for row in tickets:
        try:
            ticket = extractor.invoke(
                "Extract category, priority, product, sentiment, and a one-sentence "
                f"summary from this support ticket:\n{row['raw_text']}",
                config=config.get_run_config(prompt_version="ticket-extract-v1"),
            )
            validated = Ticket.model_validate(ticket)
            valid_count += 1
            results.append(
                {
                    "ticket_id": row["ticket_id"],
                    **validated.model_dump(),
                    "validation_status": "valid",
                }
            )
        except Exception as error:
            logger.exception("Ticket extraction failed for %s", row["ticket_id"])
            results.append(
                {
                    "ticket_id": row["ticket_id"],
                    "category": "",
                    "priority": "",
                    "product": "",
                    "sentiment": "",
                    "summary": "",
                    "validation_status": f"error: {type(error).__name__}",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    pass_rate = valid_count / len(tickets) * 100
    config.log_metric(
        "ticket_schema_valid_rate",
        pass_rate,
        valid=valid_count,
        total=len(tickets),
        minimum_valid=minimum_valid,
    )
    print(f"Schema-valid ticket outputs: {valid_count}/20 ({pass_rate:.1f}%)")
    print(f"Results: {output_path}")
    if valid_count < minimum_valid:
        raise AssertionError(
            f"Extraction acceptance failed: {valid_count} valid tickets; " f"minimum is {minimum_valid}."
        )
    return {"valid": valid_count, "total": len(tickets), "pass_rate": pass_rate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-valid", type=int, default=19)
    args = parser.parse_args()
    run_batch(minimum_valid=args.minimum_valid)
