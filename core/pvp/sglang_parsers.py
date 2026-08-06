"""Map a served model to its SGLang tool parser and optional chat template.

No parser (or 'auto', which forfeits for Qwen2.5) -> SGLang returns tool calls as
plain text and every PvP turn forfeits. Override with SGLANG_TOOL_CALL_PARSER.

The exact Gemma 4 and Ministral 3 checkpoints used by round-one tournaments are
base checkpoints with no upstream chat template. They use validator-owned
templates so model-prep baselines and evaluation can still call tools. Override
those templates with SGLANG_CHAT_TEMPLATE when operating a custom deployment.
"""

import json
import logging
import os


logger = logging.getLogger(__name__)

TOOL_CALL_PARSER_ENV = "SGLANG_TOOL_CALL_PARSER"
CHAT_TEMPLATE_ENV = "SGLANG_CHAT_TEMPLATE"
MODEL_ARGS_ENV = "SGLANG_MODEL_ARGS"

_BASE_MODEL_TOOL_CONFIG: dict[str, tuple[str, str]] = {
    "google/gemma-4-e2b": (
        "qwen",
        "core/pvp/chat_templates/gemma4_base_tool.jinja",
    ),
    "mistralai/ministral-3-3b-base-2512": (
        "qwen",
        "core/pvp/chat_templates/ministral3_base_tool.jinja",
    ),
}

_ROUND_ONE_FULL_WEIGHT_SERVING_FAMILIES = (
    "qwen3.5",
    "qwen3_5",
    "granite-4.1",
    "olmo-3",
    "olmo3",
    "olmo-hybrid",
    "olmo_hybrid",
    "lfm2.5",
    "lfm2_5",
    "nemotron-3",
    "nemotron_h",
)

# Ordered (family substring -> SGLang parser); first match wins, so more
# specific families precede the generic one (qwen3-coder before qwen; hermes
# before llama, since Hermes-3-Llama is hermes-format, not llama3).
# All model-prep/evaluation images use SGLang 0.5.14, where ``qwen25`` is a
# deprecated alias for ``qwen``.
_FAMILY_PARSERS: list[tuple[str, str]] = [
    ("qwen3.5", "qwen3_coder"),
    ("qwen3_5", "qwen3_coder"),
    ("qwen3-coder", "qwen3_coder"),
    ("nemotron-3", "qwen3_coder"),
    ("nemotron_h", "qwen3_coder"),
    ("olmo_hybrid", "olmo"),
    ("olmo-hybrid", "olmo"),
    ("olmo3", "olmo"),
    ("olmo-3", "olmo"),
    ("lfm2.5", "lfm2"),
    ("lfm2", "lfm2"),
    ("gemma4", "gemma4"),
    ("gemma-4", "gemma4"),
    ("granite", "qwen"),
    ("hermes", "hermes"),
    ("qwen3", "qwen"),
    ("qwen2", "qwen"),
    ("qwen", "qwen"),
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

    normalized_model_id = model_id.lower().rstrip(".")
    base_model_config = _BASE_MODEL_TOOL_CONFIG.get(normalized_model_id)
    if base_model_config:
        return base_model_config[0]

    parser = _parser_for_family(normalized_model_id)
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


def tool_chat_template_for(model_id: str) -> str | None:
    """Return an explicit SGLang chat template for a template-less base model."""
    override = os.getenv(CHAT_TEMPLATE_ENV)
    if override:
        return override.strip()

    config = _BASE_MODEL_TOOL_CONFIG.get(model_id.lower().rstrip("."))
    return config[1] if config else None


def requires_lora_merge_for_serving(model_id: str) -> bool:
    """Return whether this family cannot use SGLang's native LoRA path.

    The limited round-one catalog spans new hybrid, recurrent, and multimodal
    architectures whose native SGLang LoRA coverage is not uniform.  Serving a
    merged checkpoint keeps adapters on the same causal-LM architecture used
    during training while still using SGLang's supported full-weight paths (or
    the OLMo Hybrid Transformers fallback).
    """
    normalized_model_id = model_id.lower().rstrip(".")
    if normalized_model_id in _BASE_MODEL_TOOL_CONFIG:
        # These are multimodal wrapper checkpoints upstream, but tournaments
        # finetune their language-only submodels. Merge first so serving sees
        # the same causal-LM architecture that produced the adapter.
        return True
    return any(family in normalized_model_id for family in _ROUND_ONE_FULL_WEIGHT_SERVING_FAMILIES)


def sglang_model_args_for(model_id: str) -> str:
    """Return model-backend flags needed by a family in SGLang 0.5.14."""
    override = os.getenv(MODEL_ARGS_ENV)
    if override:
        return override.strip()
    return ""
