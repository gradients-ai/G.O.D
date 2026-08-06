"""Shared language-model loading for text training and evaluation.

Some Hugging Face repositories publish a multimodal outer configuration even
when Gradients only uses the language backbone.  Loading those repositories via
``AutoModelForCausalLM`` either selects a conditional-generation model or fails
because the outer config has no causal-LM mapping.  This module selects the
installed Transformers text-only class and remaps only the language weights.
"""

from __future__ import annotations

from typing import Any


OUTER_MULTIMODAL_CAUSAL_MODEL_TYPES = frozenset({"gemma4", "mistral3"})

_CONFIG_DOWNLOAD_KWARGS = frozenset(
    {
        "cache_dir",
        "force_download",
        "local_files_only",
        "proxies",
        "revision",
        "subfolder",
        "token",
        "trust_remote_code",
    }
)

_MISTRAL3_TEXT_KEY_MAPPING = {
    # Published Mistral 3 checkpoints use the legacy Llava-style serialization.
    r"^language_model\.model\.": "model.",
    r"^language_model\.lm_head\.": "lm_head.",
    # Also accept an outer model saved without the legacy conversion.
    r"^model\.language_model\.": "model.",
}
_GEMMA4_TEXT_KEY_MAPPING = {
    r"^model\.language_model\.": "model.",
}


def _load_config(model_name_or_path: str, model_kwargs: dict[str, Any]):
    from transformers import AutoConfig

    config_kwargs = {
        key: value
        for key, value in model_kwargs.items()
        if key in _CONFIG_DOWNLOAD_KWARGS and value is not None
    }
    return AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)


def is_outer_multimodal_causal_config(config) -> bool:
    """Whether ``config`` needs its language-only causal-LM implementation."""
    return getattr(config, "model_type", None) in OUTER_MULTIMODAL_CAUSAL_MODEL_TYPES


def causal_tokenizer_load_kwargs(config) -> dict[str, bool]:
    """Tokenizer compatibility kwargs required by a causal model config.

    Transformers otherwise selects ``MistralCommonBackend`` for Tekken files.
    That inference-only backend cannot persist or apply Jinja chat templates,
    while text training requires the regular tokenizers backend.
    """
    model_type = getattr(config, "model_type", None)
    if model_type in {"mistral3", "ministral3"}:
        return {"fix_mistral_regex": True}
    return {}


def load_causal_tokenizer(model_name_or_path: str, *, config=None, **kwargs):
    """Load a tokenizer suitable for causal-LM training and serialization."""
    from transformers import AutoTokenizer

    resolved_config = config or _load_config(model_name_or_path, kwargs)
    tokenizer_kwargs = causal_tokenizer_load_kwargs(resolved_config)
    tokenizer_kwargs.update(kwargs)
    return AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)


def _raise_for_incomplete_language_load(model_name_or_path: str, loading_info: dict[str, Any]) -> None:
    problems = {
        key: loading_info.get(key)
        for key in ("missing_keys", "mismatched_keys", "error_msgs", "conversion_errors")
        if loading_info.get(key)
    }
    if problems:
        details = ", ".join(f"{key}={value}" for key, value in problems.items())
        raise RuntimeError(f"Language weights were not loaded completely from {model_name_or_path}: {details}")


def load_causal_language_model(model_name_or_path: str, *model_args, config=None, **kwargs):
    """Load the text-only causal LM for ``model_name_or_path``.

    Gemma 4 and Mistral 3 repositories can wrap their language model in a
    multimodal outer checkpoint.  For those two outer configs, load the native
    text config/class directly and verify that every language parameter was
    populated.  Other architectures retain normal ``AutoModelForCausalLM``
    behavior.
    """
    from transformers import AutoModelForCausalLM

    resolved_config = config or _load_config(model_name_or_path, kwargs)
    model_type = getattr(resolved_config, "model_type", None)
    if model_type not in OUTER_MULTIMODAL_CAUSAL_MODEL_TYPES:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            *model_args,
            config=resolved_config,
            **kwargs,
        )

    if "key_mapping" in kwargs:
        raise ValueError("key_mapping is managed internally for multimodal outer causal checkpoints")

    if model_type == "mistral3":
        from transformers import Ministral3ForCausalLM

        model_class = Ministral3ForCausalLM
        key_mapping = _MISTRAL3_TEXT_KEY_MAPPING
    else:
        from transformers import Gemma4ForCausalLM

        model_class = Gemma4ForCausalLM
        key_mapping = _GEMMA4_TEXT_KEY_MAPPING

    return_loading_info = bool(kwargs.pop("output_loading_info", False))
    model, loading_info = model_class.from_pretrained(
        model_name_or_path,
        *model_args,
        config=resolved_config.get_text_config(),
        key_mapping=key_mapping,
        output_loading_info=True,
        **kwargs,
    )
    _raise_for_incomplete_language_load(model_name_or_path, loading_info)

    # Transformers remembers explicit weight conversions and reverses them on
    # save.  These mappings describe the source outer checkpoint only; retaining
    # them would serialize another outer-style prefix under a text-only config.
    model._weight_conversions = []
    model.config.architectures = [type(model).__name__]

    if return_loading_info:
        return model, loading_info
    return model
