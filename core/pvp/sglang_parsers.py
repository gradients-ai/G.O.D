"""Map a served model to its SGLang --tool-call-parser (by family).

No parser (or 'auto', which forfeits for Qwen2.5) -> SGLang returns tool calls as
plain text and every PvP turn forfeits. Override with SGLANG_TOOL_CALL_PARSER.
"""

import logging
import os


logger = logging.getLogger(__name__)

TOOL_CALL_PARSER_ENV = "SGLANG_TOOL_CALL_PARSER"

# Ordered (family substring -> SGLang parser); first match wins, so more
# specific families precede the generic one (qwen3-coder before qwen; hermes
# before llama, since Hermes-3-Llama is hermes-format, not llama3).
_FAMILY_PARSERS: list[tuple[str, str]] = [
    ("qwen3-coder", "qwen3_coder"),
    ("hermes", "hermes"),
    ("qwen3", "qwen25"),
    ("qwen2", "qwen25"),
    ("qwen", "qwen25"),
    ("llama", "llama3"),
    ("mixtral", "mistral"),
    ("mistral", "mistral"),
]


def tool_call_parser_for(model_id: str) -> str | None:
    """Return the SGLang tool-call-parser for model_id, or None if unmapped.

    SGLANG_TOOL_CALL_PARSER overrides the family map. An unmapped model logs a
    loud error (its tool calls won't be parsed and it will forfeit every turn)
    rather than silently picking a wrong parser.
    """
    override = os.getenv(TOOL_CALL_PARSER_ENV)
    if override:
        return override.strip()

    needle = model_id.lower()
    for substring, parser in _FAMILY_PARSERS:
        if substring in needle:
            return parser

    logger.error(
        "No SGLang tool-call-parser mapping for %r — tool calls will NOT be parsed "
        "and every turn will forfeit. Add a family mapping or set %s.",
        model_id,
        TOOL_CALL_PARSER_ENV,
    )
    return None
