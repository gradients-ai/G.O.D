# Image task architecture

## Supported base models

The content service supplies model IDs. The validator allowlists these model
types for tournament tasks:

- `black-forest-labs/FLUX.2-klein-base-4B` (`text-to-image`, `edit`)
- `Qwen/Qwen-Image-Edit-2511` (`edit`)
- `HiDream-ai/HiDream-O1-Image` (`text-to-image`, `edit`)
- `OpenMOSS-Team/MOSS-Image-Base` (`text-to-image`, `edit`)

`z-image`, `qwen-image`, and `krea2` (`text-to-image`) also appear in tournament
slots.

FLUX.2, Qwen, and MOSS train through the pinned AI Toolkit runtime. HiDream
text-to-image also uses AI Toolkit; HiDream edit uses pinned Musubi because
AI Toolkit does not expose O1 reference-image conditioning.

## Tournament structure

Each round task is one random draw of a `(model, mode)` slot. The full rules
are in the [image miner guide](miner_image.md#rounds).

- **Round 1**: one text-to-image task — Z-Image, FLUX.2, or MOSS — capped at
  30 source pairs.
- **Round 2**: one task from the slots of Qwen-Image (t2i), Krea2 (t2i),
  Qwen-Image-Edit (edit), and HiDream (t2i and edit).
- **Round 3**: one of six final slots — FLUX.2 / HiDream / MOSS edit, plus
  Krea2 / HiDream / MOSS text-to-image.
- **Boss round**: four tasks, two edit and two text-to-image, from the final
  slots other than round 3's.

Template step budgets are sized for the 0.5–1.5h competition window (about
1000 steps for fast families, 500 for Qwen/MOSS edit and HiDream edit).
Qwen-Image, Qwen-Image-Edit, and MOSS edit each get an extra +0.5h on top of
the size-scaled window. Outside tournament slots, dual-capability models use
the organic 40% edit probability.

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

Z-Image, Qwen-Image, and Krea2 use the ComfyUI denoising-MSE evaluator.
FLUX.2, Qwen-Image-Edit, HiDream, and MOSS use their pinned training backend to evaluate the native held-out
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

Scores are reproducible and every candidate sees identical noise and
timesteps. Each case is seeded from the base seed, the held-out sample's
identity, and how many times that sample has been scored, so the trainers'
shuffled visit order has no effect. Deterministic CUDA kernels are enabled,
and per-sample means use exact summation.

Edit scoring weights the region that changed between the held-out source and
target. A mask is built from their difference: blurred, thresholded and
dilated, with unchanged pixels kept at weight 0.2 so damage to the rest of
the image still costs. Samples with almost no change, or a near-global
change, are scored without a mask.

Candidate LoRAs are applied from the file the same way legacy Comfy eval
loads them: per-module ranks and alphas (missing `alpha` defaults to that
module's rank). LoRA keys that match no evaluation module, or a LoRA with no
non-zero weights, fail that candidate with a reason rather than scoring the
base model. Oversized ranks that OOM the evaluator fail that candidate.
