# Compression header fixtures

Safetensors headers (tensor name to dtype and shape, no data) that pin the packed layout `validator/evaluation/quant_structure.py` computes. The structural check needs only the layout, so no weights are stored.

## Contents

| Path | Purpose |
| --- | --- |
| `base_header.json`, `base_config.json` | `Qwen/Qwen2.5-0.5B-Instruct` as published. |
| `W4A16-sym-g128.json` | Header of that model's checkpoint in this format. |
| `W4A8fp8-sym-g128.json` | Header of that model's checkpoint in this format. |
| `W3A16-asym-g128.json` | Header of that model's checkpoint in this format. |

`tests/validator/evaluation/test_quant_structure.py` reads these files. `W2A16-asym-g64` has no recorded header: the test checks its shapes against hand-computed numbers.

The fixtures are the reference: when they and `packed_layout` disagree, `packed_layout` is corrected.
