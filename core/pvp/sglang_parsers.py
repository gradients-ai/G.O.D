"""Map a served model to its SGLang --tool-call-parser (by family).

No parser (or 'auto', which forfeits for Qwen2.5) -> SGLang returns tool calls as
plain text and every PvP turn forfeits. Override with SGLANG_TOOL_CALL_PARSER.
"""

import json
import logging
import os


logger = logging.getLogger(__name__)

TOOL_CALL_PARSER_ENV = "SGLANG_TOOL_CALL_PARSER"

# Ordered (family substring -> SGLang parser); first match wins, so more
# specific families precede the generic one (qwen3-coder before qwen; hermes
# before llama, since Hermes-3-Llama is hermes-format, not llama3).
# NOTE: recent SGLang deprecates 'qwen25' in favour of 'qwen' (auto-mapped with
# a warning for now); older versions only know 'qwen25'. Revisit this mapping
# when the eval/model-prep images bump SGLang.
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


def _parser_for_family(needle: str) -> str | None:
    for substring, parser in _FAMILY_PARSERS:
        if substring in needle:
            return parser
    return None


def _parser_from_local_config(model_dir: str) -> str | None:
    """Resolve the parser from config.json's model_type for a local weights dir.

    Opaque model ids (anonymized cache dirs, miner repos, augmented-<hash>) carry
    no family substring, but model_type survives anonymization (the scrubber only
    strips _name_or_path) and names the architecture family directly.

    Caveat: model_type is the architecture, not the finetune's tool-call format —
    a Hermes finetune reports model_type llama/mistral but speaks hermes format.
    Id-substring/override resolution must catch those first; this is a last
    resort where the alternative is forfeiting every turn.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path) as f:
            model_type = json.load(f).get("model_type", "")
    except Exception as exc:
        logger.warning("Could not read model_type from %s: %s", config_path, exc)
        return None
    parser = _parser_for_family(str(model_type).lower())
    if parser:
        logger.info("Resolved tool-call parser %r from config.json model_type=%r", parser, model_type)
    return parser


def tool_call_parser_for(model_id: str, *, log_unmapped: bool = True) -> str | None:
    """Return the SGLang tool-call-parser for model_id, or None if unmapped.

    Resolution order: SGLANG_TOOL_CALL_PARSER override, family substring in
    model_id, then config.json model_type when model_id is a local weights dir.
    An unmapped model logs a loud error (its tool calls won't be parsed and it
    will forfeit every turn) rather than silently picking a wrong parser; pass
    log_unmapped=False where None is expected and another resolver (the
    container's config.json fallback) gets the final word.
    """
    override = os.getenv(TOOL_CALL_PARSER_ENV)
    if override:
        return override.strip()

    parser = _parser_for_family(model_id.lower())
    if parser:
        return parser

    parser = _parser_from_local_config(model_id)
    if parser:
        return parser

    if log_unmapped:
        logger.error(
            "No SGLang tool-call-parser mapping for %r — tool calls will NOT be parsed "
            "and every turn will forfeit. Add a family mapping or set %s.",
            model_id,
            TOOL_CALL_PARSER_ENV,
        )
    return None
