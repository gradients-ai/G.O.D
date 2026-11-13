# Quant header fixtures

Safetensors headers (tensor name -> dtype + shape, no data) that pin the compressed-tensors pack-quantized layout
`validator/evaluation/quant_structure.py` computes. They replace a golden checkpoint: the structural check never
needs weights, only the layout.

- `base_header.json`, `base_config.json`: `Qwen/Qwen2.5-0.5B-Instruct` as published on the Hub (bf16, tied
  embeddings; hidden 896, intermediate 4864, kv 2 x 64, all multiples of 128).
- `W4A16-sym-g128.json`, `W4A8fp8-sym-g128.json`, `W3A16-asym-g128.json`: that model quantised with
  llmcompressor 0.14.0 / compressed-tensors 0.19.0 (CPU, torch 2.14.0):
  `QuantizationModifier(config_groups={"group_0": scheme}, ignore=["lm_head"])` + `oneshot(model=model, recipe=...)`,
  saved with `save_pretrained(..., save_compressed=True)`. The scheme targets `Linear` with these settings:
  int weights, `strategy="group"`, `group_size=128`, `actorder=None`, `dynamic=False`, `scale_dtype=torch.bfloat16`,
  `observer="memoryless_minmax"`, `format="pack-quantized"`; W4A8fp8 adds float8 input activations with
  `strategy="token"`, `dynamic=True`, `symmetric=True`.
- No `W2A16-asym-g64.json` is recorded: `test_quant_structure.py` checks its
  expected shapes against hand-computed numbers (2-bit packs 16 codes per int32, g64 scales and zero points) instead.

Regenerate by reading each shard's header (8-byte little-endian length, then JSON; drop `__metadata__`) and writing
`{name: {"dtype", "shape"}}` sorted by name, one tensor per line. If a new compressed-tensors release changes the layout, the fixture is
the truth: update `packed_layout`, not the fixture. What vLLM reads is checked separately, at evaluator image build
time (`ops/docker/scripts/check_quant_packed_layout.py`); compressed-tensors 0.17.0, which the image carries because
vLLM 0.31.0 pins it, writes a different (padded) W3 layout, but vLLM's loader does not use it.
