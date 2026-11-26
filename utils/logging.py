"""Structured, privacy-aware application logging."""

import contextlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path


def _config():
    import config

    return config


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]){1,3}\d{3,5}(?!\w)")
_SECRET_FIELD_RE = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|secret|access[_-]?token|"
    r"refresh[_-]?token|password|authorization|credential)(?:$|[_-])|"
    r"^token$|^token_(?:value|secret)$",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"\b(api[_-]?key|secret|token|access[_-]?token|refresh[_-]?token|"
    r"password|authorization|credential)"
    r"(\s*[:=]\s*)([^\s,;]+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]+=*", re.IGNORECASE)


def redact_sensitive_text(value: str) -> str:
    """Redact common direct identifiers and credential-shaped strings."""
    value = _EMAIL_RE.sub("[REDACTED_EMAIL]", value)
    value = _PHONE_RE.sub("[REDACTED_PHONE]", value)
    value = _SECRET_VALUE_RE.sub(r"\1\2[REDACTED_SECRET]", value)
    return _BEARER_RE.sub("Bearer [REDACTED_SECRET]", value)


def redact_sensitive_data(value: object, key: str = "") -> object:
    """Redact sensitive strings recursively while preserving container shape."""
    if key and _SECRET_FIELD_RE.search(key):
        return "[REDACTED_SECRET]"
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, dict):
        return {str(item_key): redact_sensitive_data(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_sensitive_data(item) for item in value]
    return value


class StructuredFormatter(logging.Formatter):
    """Emit redacted JSON events with consistent run and source context."""

    def format(self, record: logging.LogRecord) -> str:
        config = _config()
        relative_file = Path(record.pathname).name
        with contextlib.suppress(ValueError):
            relative_file = Path(record.pathname).resolve().relative_to(config.ROOT_DIR).as_posix()
        source_file = getattr(record, "source_file", config.COURSE_SOURCE_FILE or relative_file)
        source_parts = Path(source_file).parts
        source_module = ".".join(source_parts[:-1])
        day = getattr(record, "course_day", config.COURSE_DAY)
        if not day and source_parts and source_parts[0].startswith("day"):
            day = source_parts[0].split("_", maxsplit=1)[0]
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "event": redact_sensitive_text(record.getMessage()),
            "day_module": getattr(record, "day_module", source_module or record.name),
            "thread_id": getattr(record, "thread_id", ""),
            "tool_name": getattr(record, "tool_name", ""),
            "guardrail_triggered": bool(getattr(record, "guardrail_triggered", False)),
            "day": day,
            "source_file": source_file,
            "run_id": getattr(record, "run_id", config.COURSE_RUN_ID),
        }
        if hasattr(record, "metric_name"):
            event["metric"] = {
                "name": record.metric_name,
                "value": record.metric_value,
                "attributes": record.metric_attributes,
            }
        standard_fields = logging.makeLogRecord({}).__dict__
        event["details"] = {
            key: redact_sensitive_data(value, key)
            for key, value in record.__dict__.items()
            if key not in standard_fields and key not in {"message", "asctime"}
        }
        if record.exc_info:
            event["exception"] = redact_sensitive_text(self.formatException(record.exc_info))
        return json.dumps(event, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """Configure the root logger once with console and JSONL handlers."""
    config = _config()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if any(getattr(handler, "_course_structured", False) for handler in root_logger.handlers):
        return

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(StructuredFormatter())
    file_handler = logging.FileHandler(config.APP_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(StructuredFormatter())
    stream_handler._course_structured = True  # noqa: SLF001
    file_handler._course_structured = True  # noqa: SLF001
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger after ensuring shared structured handlers are configured."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("logger name must not be empty.")
    configure_logging()
    return logging.getLogger(name)


def get_structured_logger(module_name: str) -> logging.Logger:
    """Return a logger writing redacted JSON events to the shared app log."""
    return get_logger(module_name)
