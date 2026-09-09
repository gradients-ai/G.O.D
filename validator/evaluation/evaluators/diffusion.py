"""Production image L2: shared test prediction cases, separate captioned/empty-caption losses."""

import gc
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
from validator.evaluation.image_test_data import read_dataset


DEFAULT_STRATA = 16
DEFAULT_NOISES = 16


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
    else:
        encoder = adapter.load_vae()
    return adapter, model, clip, encoder


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
        api = HfApi()
        adapter, base, clip, encoder = load_base(api, config["repo"], family, config["comfy_root"])
        cases = []
        for item in validation:
            latent = encoder.encode(item["image"])
            scaled = base.model.process_latent_in(latent.clone()).float().cpu()
            cache = scratch / f"{item['sha256']}.pt"
            torch.save({"latent": latent, "scaled": scaled}, cache)
            cases.append({"id": item["sha256"], "caption": item["caption"], "cache": cache})
        # The VAE is shared and never part of a candidate's patches. Free it before scoring.
        del encoder, validation
        gc.collect()
        comfy.model_management.unload_all_models()
        result = {"model_params_count": sum(p.numel() for p in base.model.diffusion_model.parameters())}
        conditioning_cache = {}
        for repo in config["models"]:
            try:
                name, _ = materialize_model(api, repo, None, config["comfy_root"] / "models/loras")
                model, candidate_clip = adapter.apply_lora(base, clip, name)
                candidate_conditioning = {} if candidate_clip.patcher.patches else conditioning_cache
                vectors = {"text": [], "no_text": []}
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
                result[repo] = {
                    "eval_loss": {"text_guided_losses": vectors["text"], "no_text_losses": vectors["no_text"]},
                    "is_finetune": True,
                }
                print(f"Image prediction evaluation completed: {len(vectors['text'])} images", flush=True)
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
        "repo": os.environ.get("ORIGINAL_MODEL_REPO"), "family": os.environ.get("MODEL_TYPE"), "models": models,
        "comfy_root": Path(os.environ.get("COMFY_ROOT", "/app/validator/evaluation/ComfyUI")),
        "strata": DEFAULT_STRATA, "noises": DEFAULT_NOISES, "batch_size": 2,
    }
    if not all(config[k] for k in ("dataset", "repo", "family", "models")):
        raise SystemExit("Image evaluation requires test data, base repo, family and submitted models")
    try:
        evaluate(config, output)
    except Exception as error:
        print(f"Image evaluation setup failed ({type(error).__name__})", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
