import warnings


_TOKEN_ID_FIELDS = (
    "bos_token_id",
    "decoder_start_token_id",
    "eos_token_id",
    "pad_token_id",
)


def _copy_token_ids(source, target) -> None:
    if source is None:
        return

    for field in _TOKEN_ID_FIELDS:
        if hasattr(source, field):
            value = getattr(source, field)
            if value is not None:
                setattr(target, field, value)


def reset_invalid_generation_config(model, context: str) -> bool:
    """Replace invalid generation metadata so Transformers can save model weights.

    Newer Transformers releases reject generation configs whose sampling-only
    fields conflict with greedy decoding defaults. Model prep passes generation
    arguments explicitly, so falling back to a minimal config is safer than
    failing LoRA merge or augmented-model upload because of stale metadata.
    """
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return False

    invalid_reason = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                generation_config.validate(strict=True)
            except TypeError:
                generation_config.validate()
        if not caught:
            return False
        invalid_reason = "; ".join(str(warning.message) for warning in caught[:3])
    except Exception as exc:
        invalid_reason = str(exc)

    try:
        from transformers import GenerationConfig

        safe_config = GenerationConfig()
        _copy_token_ids(getattr(model, "config", None), safe_config)
        _copy_token_ids(generation_config, safe_config)
        model.generation_config = safe_config
        print(
            f"[trainer] Reset invalid generation_config before {context}: {invalid_reason}",
            flush=True,
        )
        return True
    except Exception as exc:
        print(
            f"[trainer] WARNING: failed to reset invalid generation_config before {context}: {exc}",
            flush=True,
        )
        return False
