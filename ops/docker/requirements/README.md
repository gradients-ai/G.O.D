# Docker Requirements

Requirements files used only inside Docker builds.

## Contents

- `image-evaluator.txt`: image evaluator (`validator-diffusion.dockerfile`) Python requirements.
- `text-evaluator-pins.in`: direct requirements of the text evaluator's numerics stack - the axolotl `0bda5a13` pins the
  eval imports (transformers 5.12.1, torch 2.9.1, trl 1.5.1, accelerate 1.13.0, peft 0.19.x, datasets 4.8.x, ...), the
  hybrid/linear-attention kernels the old axolotl base shipped, and build tools.
- `text-evaluator-pins.txt`: the fully pinned lock compiled from `text-evaluator-pins.in`; `validator.dockerfile` installs it
  with `uv pip sync` and constrains every later install with it.
- `validator.txt`: text evaluator image application requirements (installed on top of the pins, constrained by them).

## Regenerating the text-evaluator pins

The pins reproduce the stack of the image the evaluator used to be built `FROM`
(`axolotlai/axolotl:main-20260701-py3.11-cu128-2.9.1`), so the `--exclude-newer` date stays the day after that tag's
build date; only change it together with a deliberate numerics upgrade (which moves every text eval loss).

```bash
cd ops/docker/requirements
uv pip compile --python-version 3.11 --python-platform x86_64-manylinux_2_28 --torch-backend cu128 \
  --exclude-newer 2026-07-02T00:00:00Z --no-header text-evaluator-pins.in -o text-evaluator-pins.txt
```

Then restore the explanatory header at the top of `text-evaluator-pins.txt`. When a docker daemon is available, cross-check
against the old base with `docker run --rm axolotlai/axolotl:main-20260701-py3.11-cu128-2.9.1 uv pip freeze`: any difference
in transformers / torch / trl / peft / accelerate / datasets / tokenizers is a blocker.
