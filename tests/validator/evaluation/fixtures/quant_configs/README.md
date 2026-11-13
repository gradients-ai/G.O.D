# Quant config fixtures

Model configs that pin what `config_drift_failure` (`validator/evaluation/quant_config_check.py`) must accept: an
honest fixed-track save whose config.json was re-serialised by transformers 5. Each file holds four configs:

- `input_model_raw`: the input model's config.json as published on the Hub.
- `input_model`: that config resolved, `AutoConfig.from_pretrained(repo).to_dict()` with transformers 5.17.0 (the
  quant evaluator image's pin; vLLM 0.31.0's `get_config` resolves these model types the same way, then applies the
  same RoPE patch to both sides).
- `honest_save_raw`: the config.json a bf16 W4A16 save writes when transformers serialises it: `dtype="bfloat16"`,
  `architectures` set from the model class (what `PreTrainedModel.save_pretrained` sets), `config.save_pretrained`,
  then a `quantization_config` added (abbreviated to `quant_method` and `format`; the spec check is tested on its
  own). llm-compressor 0.14 normally restores the original file instead (`resave_config`), but keeps this
  transformers-serialised form whenever the original lacks `tie_word_embeddings` or `dtype`/`torch_dtype`, as Gemma
  configs do.
- `honest_save`: `honest_save_raw` resolved the same way, with `_name_or_path` set to a fixed path.

Models:

- `olmo2_1b.json`: `allenai/OLMo-2-0425-1B`, published as `torch_dtype: float32`, so the bf16 save changes the dtype.
- `gemma3_1b.json`: `unsloth/gemma-3-1b-pt`; transformers 5 rewrites `rope_theta`/`rope_local_base_freq` into nested
  `rope_parameters`, adds `layer_types`, `_sliding_window_pattern`, `tie_word_embeddings` and renames `torch_dtype`.

Regenerate with transformers 5.17.0 by repeating the steps above for each repo (the raw configs come from
`hf_hub_download(repo, "config.json")`).
