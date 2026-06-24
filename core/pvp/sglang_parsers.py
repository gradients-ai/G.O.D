"""Map a served model to its SGLang --tool-call-parser.

No parser (or 'auto', which forfeits for Qwen2.5) -> SGLang returns tool calls as
plain text and every PvP turn forfeits. Override with SGLANG_TOOL_CALL_PARSER.

Resolution prefers the model's chat template (the actual contract for how tool
calls are rendered) over name/architecture proxies, because both the repo name
and config.json model_type lie once a model is anonymized/augmented: an opaque
'augmented-<hash>' id carries no family substring, and a Hermes finetune reports
model_type 'llama'/'mistral' while speaking hermes tool-call format. The chat
template survives anonymization (the scrubber only strips _name_or_path) and
names the format directly.
"""

import functools
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

# Ordered (chat-template marker -> SGLang parser); first match wins. These are
# the distinctive tokens each format emits for tool calls, so the match is the
# format itself rather than a proxy for it. Qwen2.5/3 adopted the Hermes
# <tool_call> JSON wrapper, and SGLang's hermes parser handles that wrapper, so
# both map to 'hermes' here. qwen3-coder uses a distinct <function= ...> body
# inside <tool_call> and must be checked first.
_TEMPLATE_MARKERS: list[tuple[str, str]] = [
    ("[TOOL_CALLS]", "mistral"),
    ("<|python_tag|>", "llama3"),
    ("<function=", "qwen3_coder"),
    ("<tool_call>", "hermes"),
]

# Files that may carry the chat template, in priority order. Newer HF layouts
# split the template into a standalone chat_template.jinja; older ones inline it
# under tokenizer_config.json's "chat_template" key.
_CHAT_TEMPLATE_FILE = "chat_template.jinja"
_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


def _parser_for_family(needle: str) -> str | None:
    for substring, parser in _FAMILY_PARSERS:
        if substring in needle:
            return parser
    return None


def _parser_from_local_config(model_dir: str) -> str | None:
    """Resolve the parser from config.json's model_type for a local weights dir.

    Last resort: model_type is the architecture, not the finetune's tool-call
    format, so a Hermes finetune reports llama/mistral here. The chat-template
    and id-substring resolvers must run first.
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


def _parser_for_template(template: str) -> str | None:
    for marker, parser in _TEMPLATE_MARKERS:
        if marker in template:
            return parser
    return None


def _chat_template_from_config(raw: str) -> str | None:
    """Pull the chat template out of a tokenizer_config.json blob.

    The "chat_template" value is usually a Jinja string, but some repos ship a
    list of {name, template} dicts (e.g. a separate 'tool_use' template); join
    them so a tool template anywhere in the list is still matched.
    """
    try:
        chat_template = json.loads(raw).get("chat_template")
    except Exception as exc:
        logger.warning("Could not parse tokenizer config for chat_template: %s", exc)
        return None
    if isinstance(chat_template, str):
        return chat_template
    if isinstance(chat_template, list):
        return "\n".join(t["template"] for t in chat_template if isinstance(t, dict) and "template" in t)
    return None


def _read_local_chat_template(model_dir: str) -> str | None:
    jinja_path = os.path.join(model_dir, _CHAT_TEMPLATE_FILE)
    if os.path.isfile(jinja_path):
        try:
            with open(jinja_path) as f:
                return f.read()
        except Exception as exc:
            logger.warning("Could not read %s: %s", jinja_path, exc)
    config_path = os.path.join(model_dir, _TOKENIZER_CONFIG_FILE)
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                return _chat_template_from_config(f.read())
        except Exception as exc:
            logger.warning("Could not read %s: %s", config_path, exc)
    return None


def _read_hub_chat_template(repo_id: str) -> str | None:
    """Download just the chat template for a remote HF repo (tiny file).

    Returns None on any failure (missing file, private repo, network) — the
    caller falls through to cheaper resolvers, and an unmapped model still logs
    loudly. Imported lazily so non-eval call sites don't pay the import.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    token = os.getenv("HF_TOKEN")
    try:
        path = hf_hub_download(repo_id, _CHAT_TEMPLATE_FILE, token=token)
        with open(path) as f:
            return f.read()
    except EntryNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Could not fetch %s from %s: %s", _CHAT_TEMPLATE_FILE, repo_id, exc)
        return None
    try:
        path = hf_hub_download(repo_id, _TOKENIZER_CONFIG_FILE, token=token)
        with open(path) as f:
            return _chat_template_from_config(f.read())
    except Exception as exc:
        logger.warning("Could not fetch %s from %s: %s", _TOKENIZER_CONFIG_FILE, repo_id, exc)
        return None


@functools.lru_cache(maxsize=128)
def _parser_from_chat_template(model_id: str) -> str | None:
    """Resolve the parser from model_id's chat template (local dir or HF repo).

    Cached because the same served model is resolved more than once per eval and
    the remote case hits the network.
    """
    if os.path.isdir(model_id):
        template = _read_local_chat_template(model_id)
    else:
        template = _read_hub_chat_template(model_id)
    if not template:
        return None
    parser = _parser_for_template(template)
    if parser:
        logger.info("Resolved tool-call parser %r from chat template of %r", parser, model_id)
    return parser


def tool_call_parser_for(model_id: str, *, log_unmapped: bool = True) -> str | None:
    """Return the SGLang tool-call-parser for model_id, or None if unmapped.

    Resolution order: SGLANG_TOOL_CALL_PARSER override, family substring in
    model_id (cheap, correct for clearly-named repos), the model's chat template
    (the real tool-call-format contract, and the only signal that survives
    anonymization/augmentation), then config.json model_type for a local dir.
    An unmapped model logs a loud error (its tool calls won't be parsed and it
    will forfeit every turn) rather than silently picking a wrong parser; pass
    log_unmapped=False where None is expected and another resolver gets the
    final word.
    """
    override = os.getenv(TOOL_CALL_PARSER_ENV)
    if override:
        return override.strip()

    parser = _parser_for_family(model_id.lower())
    if parser:
        return parser

    parser = _parser_from_chat_template(model_id)
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
