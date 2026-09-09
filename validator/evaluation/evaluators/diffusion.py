"""Production image L2: shared held-out prediction cases, 50/50 captioned/empty caption."""

import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

from validator.evaluation.denoising_mse import flow_cases
from validator.evaluation.denoising_mse import flow_prediction_mse
from validator.evaluation.image_artifacts import atomic_json
from validator.evaluation.image_artifacts import materialize_model
from validator.evaluation.image_artifacts import prepare_base
from validator.evaluation.image_denoising import FlowPredictionSampler
from validator.evaluation.image_encoder import DeterministicVAE
from validator.evaluation.image_flow_adapter import FAMILIES
from validator.evaluation.image_flow_adapter import ImageFlowAdapter
from validator.evaluation.image_test_data import decoded_images
from validator.evaluation.image_test_data import held_out_images
from validator.evaluation.image_test_data import read_dataset


VERSION = "image-denoising-l2-v1"
TEXT_WEIGHT = 0.5
DEFAULT_STRATA = 16
DEFAULT_NOISES = 16


def positive_env(name, default):
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def scalar_score(text, no_text):
    if not text or len(text) != len(no_text) or any(not math.isfinite(v) or v < 0 for v in [*text, *no_text]):
        raise ValueError("Aligned nonempty finite nonnegative image losses required")
    return TEXT_WEIGHT * float(np.mean(text)) + (1 - TEXT_WEIGHT) * float(np.mean(no_text))


def load_base(api, repo, family, root):
    """Load the task checkpoint with the validated family contract."""
    import nodes

    adapter = ImageFlowAdapter(family, api, root)
    name, provenance = prepare_base(api, {"model_id": repo, "model_type": family}, root)
    model = adapter.configure(nodes.UNETLoader().load_unet(name, "default")[0])
    clip = adapter.load_clip()
    if family == "z-image":
        from diffusers import AutoencoderKL

        encoder = DeterministicVAE(AutoencoderKL.from_pretrained(
            repo, subfolder="vae", revision=provenance["revision"], torch_dtype=torch.float32
        ))
        adapter.provenance["evaluation_vae"] = {"repo": repo, "revision": provenance["revision"], "subfolder": "vae"}
    else:
        encoder = adapter.load_vae()
    return adapter, model, clip, encoder, {"base": provenance, "adapter": adapter.provenance}


@torch.inference_mode()
def evaluate(config, output):
    from huggingface_hub import HfApi

    family = config["family"]
    if family not in FAMILIES:
        raise ValueError("Unsupported image model family")
    sys.path.insert(0, str(config["comfy_root"]))
    import comfy.model_management
    import comfy.samplers

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="image-l2-") as scratch:
        scratch = Path(scratch)
        validation = decoded_images(read_dataset(config["dataset"], scratch / "test.zip"))
        training = decoded_images(read_dataset(config["training_dataset"], scratch / "train.zip", require_captions=False))
        validation, excluded = held_out_images(training, validation)
        # Bound only manual/container smoke runs. Production evaluates every retained test image.
        if config.get("max_images"):
            validation = validation[:config["max_images"]]
        del training
        api = HfApi()
        adapter, base, clip, encoder, provenance = load_base(api, config["repo"], family, config["comfy_root"])
        kind = "flow_prediction_mse"
        cases = []
        for item in validation:
            latent = encoder.encode(item["image"])
            scaled = base.model.process_latent_in(latent.clone()).float().cpu()
            cache = scratch / f"{item['sha256']}.pt"
            torch.save({"latent": latent, "scaled": scaled}, cache)
            cases.append({"id": item["sha256"], "caption": item["caption"], "cache": cache,
                          "latent_hash": hashlib.sha256(scaled.numpy().tobytes()).hexdigest()})
        # Hash tensor content independently of torch archive filenames/serialization.
        fingerprint_payload = {
            "version": VERSION, "metric": kind, "family": family, "provenance": provenance,
            "image_ids": [c["id"] for c in cases],
            "latent_hashes": [c["latent_hash"] for c in cases],
            "captions": [hashlib.sha256(c["caption"].encode()).hexdigest() for c in cases],
            "strata": config["strata"], "noises": config["noises"], "batch_size": config["batch_size"],
            "text_weight": TEXT_WEIGHT, "seed": 42,
            "comfy_commit": (config["comfy_root"] / ".git/HEAD").read_text().strip(),
            "torch": torch.__version__,
        }
        fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
        # The VAE is shared and never part of a candidate's patches. Free it before scoring.
        del encoder, validation
        gc.collect()
        comfy.model_management.unload_all_models()
        result = {"model_params_count": sum(p.numel() for p in base.model.diffusion_model.parameters())}
        conditioning_cache = {}
        for repo in config["models"]:
            try:
                name, artifact = materialize_model(api, repo, None, config["comfy_root"] / "models/loras")
                model, candidate_clip = adapter.apply_lora(base, clip, name)
                candidate_conditioning = {} if candidate_clip.patcher.patches else conditioning_cache
                vectors = {"text": [], "no_text": []}
                per_case = []
                first_check = None
                for item in cases:
                    data = torch.load(item["cache"], map_location="cpu", weights_only=True)
                    questions = flow_cases(
                        data["scaled"], item["id"], config["strata"], config["noises"]
                    )
                    for mode in vectors:
                        prompt = item["caption"] if mode == "text" else ""
                        if prompt not in candidate_conditioning:
                            candidate_conditioning[prompt] = adapter.conditioning(candidate_clip, prompt)
                        sampler = FlowPredictionSampler(questions, config["batch_size"], flow_prediction_mse)
                        guider = comfy.samplers.CFGGuider(model)
                        guider.set_conds(candidate_conditioning[prompt], [])
                        guider.set_cfg(1.0)
                        schedule = torch.tensor([q["sigma"] for q in reversed(questions)] + [0.0])
                        guider.sample(torch.zeros_like(data["latent"]), data["latent"], sampler, schedule,
                                      disable_pbar=True, seed=42)
                        losses = [row["mse"] for row in sampler.rows]
                        if len(losses) != config["strata"] * config["noises"]:
                            raise ValueError("Incomplete prediction cases")
                        vectors[mode].append(float(np.mean(losses)))
                        per_case.extend({**row, "image_id": item["id"], "mode": mode} for row in sampler.rows)
                        if first_check is None:
                            first_check = (data, questions, candidate_conditioning[prompt], losses)
                data, questions, cond, original = first_check
                repeated = FlowPredictionSampler(questions, config["batch_size"], flow_prediction_mse)
                guider = comfy.samplers.CFGGuider(model)
                guider.set_conds(cond, [])
                guider.set_cfg(1.0)
                schedule = torch.tensor([q["sigma"] for q in reversed(questions)] + [0.0])
                guider.sample(torch.zeros_like(data["latent"]), data["latent"], repeated, schedule, disable_pbar=True, seed=42)
                if not np.allclose(original, [r["mse"] for r in repeated.rows], rtol=1e-5, atol=1e-7):
                    raise ValueError("Prediction scores failed repeatability check")
                score = scalar_score(vectors["text"], vectors["no_text"])
                result[repo] = {
                    "eval_loss": score, "is_finetune": True, "metric_version": VERSION,
                    "metric": kind, "text_weight": TEXT_WEIGHT, "eval_set_fingerprint": fingerprint,
                    "text_guided_losses": vectors["text"], "no_text_losses": vectors["no_text"],
                    "image_ids": [c["id"] for c in cases], "strata": config["strata"], "noises": config["noises"],
                }
                if config.get("audit_dir"):
                    audit = config["audit_dir"]
                    audit.mkdir(parents=True, exist_ok=True)
                    atomic_json(audit / (hashlib.sha256(repo.encode()).hexdigest() + ".json"), {
                        "result": result[repo], "cases": per_case, "artifact": artifact,
                        "provenance": provenance, "excluded_count": len(excluded),
                    })
                print(f"Image prediction evaluation completed: {len(vectors['text'])} images, loss={score:.6f}", flush=True)
            except Exception as error:
                # Exceptions from remote downloads can contain signed URLs; never serialize their message.
                result[repo] = f"Image prediction evaluation failed ({type(error).__name__})"
                print(result[repo], flush=True)
                for frame in traceback.extract_tb(error.__traceback__):
                    print(f"  {Path(frame.filename).name}:{frame.lineno} in {frame.name}", flush=True)
                if isinstance(error, RuntimeError):
                    message = str(error).splitlines()[0]
                    if message.startswith(("mat1 and mat2", "expected scalar type", "Input type", "The size of tensor", "shape '",
                                           "CUDA out of memory", "Expected all tensors", "Expected tensor")):
                        print(message, flush=True)
            finally:
                atomic_json(output, result)
                comfy.model_management.unload_all_models()
                gc.collect()
        return result


def main():
    output = Path(os.environ.get("EVALUATION_RESULTS_PATH", "/aplp/evaluation_results.json"))
    models = [m.strip() for m in os.environ.get("MODELS", "").split(",") if m.strip()]
    config = {
        "dataset": os.environ.get("DATASET") or os.environ.get("TEST_SPLIT_URL"),
        "training_dataset": os.environ.get("TRAINING_DATASET") or os.environ.get("TRAIN_SPLIT_URL"),
        "repo": os.environ.get("ORIGINAL_MODEL_REPO"), "family": os.environ.get("MODEL_TYPE"), "models": models,
        "comfy_root": Path(os.environ.get("COMFY_ROOT", "/app/validator/evaluation/ComfyUI")),
        "strata": positive_env("IMAGE_EVAL_STRATA", DEFAULT_STRATA),
        "noises": positive_env("IMAGE_EVAL_NOISES", DEFAULT_NOISES),
        "batch_size": positive_env("IMAGE_EVAL_NOISE_BATCH_SIZE", 2),
        "max_images": int(os.environ.get("IMAGE_EVAL_MAX_IMAGES", "0")),
        "audit_dir": Path(os.environ["IMAGE_EVAL_AUDIT_DIR"]) if os.environ.get("IMAGE_EVAL_AUDIT_DIR") else None,
    }
    if not all(config[k] for k in ("dataset", "training_dataset", "repo", "family", "models")) or config["max_images"] < 0:
        raise SystemExit("Image evaluation requires test data, training data, base repo, family and submitted models")
    try:
        evaluate(config, output)
    except Exception as error:
        print(f"Image evaluation setup failed ({type(error).__name__})", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
