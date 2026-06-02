# Training Templates

Base config files used by trainer entrypoints to generate per-task training configs.

## Contents

- `base.yml`: default Axolotl text training template.
- `base_grpo.yml`: Axolotl GRPO template.
- `base_environment.yml`: environment-task training template.
- `base_diffusion_sdxl.toml`: SDXL image training template.
- `base_diffusion_flux.toml`: Flux image training template.
- `base_diffusion_zimage.yaml`: Z-Image training template.
- `base_diffusion_qwen_image.yaml`: Qwen image training template.

These files are runtime inputs, not examples. Keep path assumptions aligned with `trainer/constants.py` and `trainer/training_paths.py`.
