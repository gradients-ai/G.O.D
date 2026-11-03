# Image task architecture

## Supported base models

The content service supplies model IDs. The validator allowlists these model
types for tournament tasks:

- `black-forest-labs/FLUX.2-klein-base-4B` (`text-to-image`, `edit`)
- `Qwen/Qwen-Image-Edit-2511` (`edit`)
- `HiDream-ai/HiDream-O1-Image` (`text-to-image`, `edit`)
- `OpenMOSS-Team/MOSS-Image-Base` (`text-to-image`, `edit`)

FLUX.2, Qwen, and MOSS train through the pinned AI Toolkit runtime. HiDream
text-to-image also uses AI Toolkit; HiDream edit uses pinned Musubi because
AI Toolkit does not expose O1 reference-image conditioning.

## Image-edit dataset contract

An edit sample contains:

- a source image (`source_image_url`);
- a target image (`image_url`);
- a natural-language edit instruction (`text_url`).

Task preparation writes a `manifest.jsonl` with one source per target and
keeps complete source/target/instruction groups together during the held-out
split. Trainer containers reject missing controls and paths that escape the
dataset root.

Synthetic edit tasks first ask the prompt LLM to invent one reusable edit
objective. It then proposes diverse source scenes and objective-specific
instructions. FAL generates each source and applies the edit to produce its
paired target. The objective stays fixed within a task while subject, pose,
framing, lighting, scene, and palette vary.

Product and edit prompt generation receive a randomized creative brief,
run at a higher temperature, exclude common product defaults, and keep a
bounded recent-concept history. Near-duplicate concepts are rejected and
regenerated.

## Evaluation

Existing image families keep the ComfyUI denoising-MSE evaluator. The four
new families use their pinned training backend to evaluate the native held-out
training objective: AI Toolkit for supported text-to-image/edit paths and
Musubi for HiDream O1 editing. Conditional and empty-text passes reuse the
same deterministic seeds and preserve GOD's existing `DiffusionLosses` result
contract and remote per-repository evaluation flow. Generated-image pixel
comparison is not used for ranking.

AI Toolkit and Musubi share one evaluator image but use isolated Python
environments. This keeps their incompatible pinned Transformers and Diffusers
versions from replacing each other while avoiding a Dockerfile per backend.

Scoring uses the noisiest training-objective band so easy late-denoising
steps cannot collapse the ranking signal. AI Toolkit families sample
`min_denoising_steps=0` / `max_denoising_steps=250` (t ≈ 750–1000). HiDream
is tighter because it predicts pixel-space x0 with `noise_scale=8`: T2I
`max_denoising_steps=50`, edit Musubi `--min_timestep 950`.

Evaluation defaults to four held-out examples per candidate, ten repeats,
and fixed seeds. `IMAGE_OBJECTIVE_EVAL_SAMPLES`, `IMAGE_OBJECTIVE_EVAL_REPEATS`,
and `IMAGE_OBJECTIVE_EVAL_SEED` can override those values.
