import json
import logging
import os
import re
import sys


_BASILICA_LOG_LINE_OFFSETS: dict[str, int] = {}


def configure_eval_logging() -> None:
    """Configure root logger for eval containers (stderr, replaces existing handlers)."""
    level_name = os.getenv("EVAL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(level)
    for existing in root.handlers[:]:
        root.removeHandler(existing)
        try:
            existing.close()
        except Exception:
            pass
    root.addHandler(handler)

def _log_eval_step(eval_logger: logging.Logger, step: str, **fields) -> None:
    field_text = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    eval_logger.info(f"eval_step={step} {field_text}".rstrip())

def clean_basilica_log_line(raw_line: str) -> str:
    line = raw_line.strip()
    if not line:
        return ""
    line = re.sub(r"^data:\s*", "", line).rstrip(", ")
    for _ in range(2):
        try:
            parsed = json.loads(line)
        except Exception:
            break

        if isinstance(parsed, dict):
            extracted = parsed.get("message") or parsed.get("log") or parsed.get("data")
            if isinstance(extracted, str) and extracted.strip():
                line = extracted.strip()
                continue
            line = str(parsed)
            break

        if isinstance(parsed, str):
            line = parsed.strip()
            continue

        line = str(parsed)
        break
    if "\\u001b" in line or "\\x1b" in line:
        try:
            line = bytes(line, "utf-8").decode("unicode_escape")
        except Exception:
            pass

    line = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", line)
    line = re.sub(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?\]\s*", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line

def log_basilica_logs_block(eval_logger: logging.Logger, repo: str, deployment_name: str, deployment) -> None:
    try:
        raw_logs = deployment.logs()
    except Exception as e:
        eval_logger.warning(f"[BASILICA_LOG_FETCH_FAILED] repo={repo} deployment={deployment_name} error={e}")
        return

    if not raw_logs:
        eval_logger.info(f"[BASILICA_LOGS] repo={repo} deployment={deployment_name} lines=0 message=\"no logs returned\"")
        return

    if isinstance(raw_logs, bytes):
        raw_logs = raw_logs.decode("utf-8", errors="replace")

    lines = []
    for raw_line in str(raw_logs).splitlines():
        cleaned = clean_basilica_log_line(raw_line)
        if cleaned:
            lines.append(cleaned)

    if not lines:
        eval_logger.info(
            f"[BASILICA_LOGS] repo={repo} deployment={deployment_name} lines=0 "
            "message=\"log payload present but no parsable lines\""
        )
        return

    previous_count = _BASILICA_LOG_LINE_OFFSETS.get(deployment_name, 0)
    if previous_count > len(lines):
        previous_count = 0
    new_lines = lines[previous_count:]
    _BASILICA_LOG_LINE_OFFSETS[deployment_name] = len(lines)

    if not new_lines:
        eval_logger.info(
            f"[BASILICA_LOGS] repo={repo} deployment={deployment_name} new_lines=0 total_lines={len(lines)}"
        )
        return

    eval_logger.info(
        f"[BASILICA_LOGS] repo={repo} deployment={deployment_name} "
        f"new_lines={len(new_lines)} total_lines={len(lines)}"
    )
    for line_number, line in enumerate(new_lines, start=previous_count + 1):
        eval_logger.info(f"[BASILICA_LOG] repo={repo} deployment={deployment_name} line={line_number} | {line}")
