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
split.

Synthetic image tasks generate `IMAGE_SYNTH_TEST_PAIRS` (8) pairs on top of the
training pairs they draw, and the split holds out exactly those: the generator
writes them last, so they are the highest-numbered pairs. They keep the task's
concept but put it where training never goes (a product in use or at night, a
design system on a watch or billboard, a mascot drawn in the trained style, an edit
rule on unseen object types), so a LoRA that memorised its training set scores
badly. The training set keeps its size and training hours scale with the training
pairs. Organic datasets keep the random 10% split.

Each synthetic task locks one concept ([`synthetics/concepts.py`](../validator/tasks/synthetics/concepts.py)):
an LLM writes a detailed identity spec and short contexts, a reference image is
generated and checked against the spec by a vision model, and every image is
rendered from that reference and checked against it again. Captions are only the
concept's name plus the context; the spec reaches the image generator, never the
caption, so the concept must be learned from the images. Edit tasks name a rule
only ("Apply the X finish."). Categories are weighted by
`IMAGE_SYNTH_CATEGORY_WEIGHTS`. Trainer containers reject missing controls and paths that escape the
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
same deterministic seeds and return the same `DiffusionLosses` result
contract and remote per-repository evaluation flow. Generated-image pixel
comparison is not used for ranking.

AI Toolkit and Musubi share one evaluator image but use isolated Python
environments. This keeps their incompatible pinned Transformers and Diffusers
versions from replacing each other while avoiding a Dockerfile per backend.

Every image family is scored on plain L2 in the band sigma 0.70-0.95. Above 0.95 the
prediction is close to the shared noise floor and rankings flip between evaluation
seeds; below 0.70 most of the image is visible and an untrained base model catches up
with trained LoRAs. Inside the band the trained LoRAs are ordered the same way on every
seed and the base model is clearly worst.

Every task is scored unguided: 0.5 x the captioned loss + 0.5 x the empty-prompt loss
(`DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT`). A classifier-free-guided prediction is not used:
at this noise its error grows with how strongly the LoRA commits to what it learned, so
on held-out compositions it ranks an untrained base model above LoRAs that learned the
task. The empty-prompt loss measures how much of the task the LoRA learned; the
captioned loss measures how well it follows each prompt. Edit tasks weight the edited
region and the rest of the image equally (`MASK_TARGET_CHANGED_MASS = 0.5`), so a
candidate must make the edit and must not damage what it should leave alone.

| Family | Noise levels | Cases |
| --- | --- | --- |
| Z-Image, Qwen-Image, Krea2 (ComfyUI) | 8 even strata in sigma 0.70-0.95 | about 512 per pass whatever the test size |
| FLUX.2 (AI Toolkit), text-to-image and edit | sigma 0.70-0.95 (`min/max_denoising_steps=50/300`) | about 240 per pass: up to 8 held-out examples x 10-40 repeats |
| MOSS, Qwen-Image-Edit (AI Toolkit), text-to-image and edit | as above | about 120 per pass (MOSS is about four times slower per case) |
| HiDream O1 (AI Toolkit), text-to-image | sigma 0.54-0.84 (`min/max_denoising_steps=160/460`) | as FLUX.2 |
| HiDream O1 (Musubi), edit | sigma 0.33-1.0 (`--min_timestep 330`) | as FLUX.2 |

HiDream O1 noises pixel-space x0 with `noise_scale=8`, so its SNR at sigma is
(1 - sigma) / (8 sigma) and the difficulty of latent sigma 0.9 sits near sigma 0.53.
Above sigma 0.84 its input is close to pure noise with nothing left to denoise.

On the AI Toolkit path the cases of each held-out image are spread evenly across the
band and come in antithetic pairs: repeats 2k and 2k+1 share one noise draw with its
sign flipped, and pair k takes the k-th equal slice of the band
([`stratified_case`](../validator/evaluation/objective_seeding.py)). This removes most
of the variance that comes from which noise levels and noise samples a seed happens to
draw. The captioned and empty-prompt passes see identical noise and timesteps for each
case. On the ComfyUI path the strata are fixed levels, each with several noise draws.

Image L2 differences between trained LoRAs are tenths of a percent, consistently
across seeds. The crown margin is set just above the gap between two trainings of the
same repository: a challenger must beat the boss's loss by `IMAGE_CROWN_WIN_MARGIN`
(0.15%) on a task to win it.

Per-candidate compute is held fixed so a bigger test split does not add time.
`IMAGE_OBJECTIVE_EVAL_SAMPLES`, `IMAGE_OBJECTIVE_EVAL_REPEATS`, `IMAGE_EVAL_STRATA`
and `IMAGE_EVAL_NOISES` override the defaults.

Every image evaluation is seeded from the task's `eval_seed`. The validator
draws it at random on the task's first evaluation, stores it in
`image_tasks.eval_seed` and sends it to every deployment as `IMAGE_EVAL_SEED`,
which both evaluators require. So every candidate and every retry on a task sees
identical noise and timesteps, and the noise cannot be derived before the task
is evaluated. The seed is hidden while the task is in flight and published by
the auditing endpoints once it finishes.

Each case is seeded from the task seed, a hash of the held-out sample's image
bytes, and how many times that sample has been scored, so neither the test
images' order nor the trainers' shuffled visit order has an effect. Deterministic CUDA kernels are enabled,
and per-sample means use exact summation.

Edit scoring weights the region that changed between the held-out source and
target. A mask is built from their difference: blurred, thresholded and
dilated, with unchanged pixels kept at weight 0.2 so damage to the rest of
the image still costs. Samples with almost no change, or a near-global
change, are scored without a mask.

Candidate LoRAs are applied from the file the same way the ComfyUI evaluator
loads them: per-module ranks and alphas (missing `alpha` defaults to that
module's rank). LoRA keys that match no evaluation module, or a LoRA with no
non-zero weights, fail that candidate with a reason rather than scoring the
base model. Oversized ranks that OOM the evaluator fail that candidate.
