"""Score a HiDream O1 edit LoRA with Musubi's native training objective, without updating it.

Launched with Musubi's ``accelerate``; extends ``HiDreamO1NetworkTrainer`` rather
than patching it. Configuration that Musubi's CLI does not know comes from env:
GOD_EVAL_CACHE_DIR, GOD_EVAL_OUTPUT, GOD_EVAL_SEED and optional GOD_EVAL_MASK_DIR.
"""

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import torch
from musubi_tuner.hidream_o1 import hidream_o1_utils
from musubi_tuner.hidream_o1_train_network import HiDreamO1NetworkTrainer
from musubi_tuner.hidream_o1_train_network import hidream_o1_setup_parser
from musubi_tuner.hv_train_network import read_config_from_file
from musubi_tuner.hv_train_network import setup_parser_common
from musubi_tuner.utils.model_utils import remove_dtype_suffix
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from validator.evaluation.edit_masks import cover_and_center_crop
from validator.evaluation.edit_masks import mask_weights
from validator.evaluation.objective_seeding import CaseSeeder
from validator.evaluation.objective_seeding import enable_deterministic_kernels


def _fingerprint(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _latent_fingerprints(cache_dir: Path) -> dict[str, str]:
    fingerprints = {}
    for path in sorted(cache_dir.glob("*.safetensors")):
        if path.name.endswith("_te.safetensors"):
            continue
        with safe_open(str(path), framework="pt") as handle:
            keys = [
                key
                for key in handle.keys()
                if key.startswith("latents_") and remove_dtype_suffix(key).rsplit("_", 1)[0] == "latents"
            ]
            if len(keys) != 1:
                raise ValueError(f"Expected one target latent in {path.name}; found {len(keys)}")
            fingerprint = _fingerprint(handle.get_tensor(keys[0]))
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate held-out sample content in {path.name}")
        fingerprints[fingerprint] = path.name.split("_", 1)[0]
    if not fingerprints:
        raise ValueError(f"No Musubi latent caches in {cache_dir}")
    return fingerprints


def _weights_with_default_alphas(path: str) -> dict[str, torch.Tensor]:
    """Fill missing .alpha keys so create_network_from_weights accepts mixed-rank LoRAs."""
    weights_sd = load_file(path)
    for key, value in list(weights_sd.items()):
        if "lora_down" not in key or not key.endswith(".weight"):
            continue
        alpha_key = f"{key.split('.')[0]}.alpha"
        if alpha_key not in weights_sd:
            weights_sd[alpha_key] = torch.tensor(float(value.shape[0]))
    return weights_sd


class GODHiDreamObjectiveEvaluator(HiDreamO1NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.output = Path(os.environ["GOD_EVAL_OUTPUT"])
        self.seeder = CaseSeeder(int(os.environ["GOD_EVAL_SEED"]))
        self.sample_keys = _latent_fingerprints(Path(os.environ["GOD_EVAL_CACHE_DIR"]))
        mask_dir = os.environ.get("GOD_EVAL_MASK_DIR")
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.records = []
        self.current_key = None

    def _build_network(self, args, accelerator, transformer, vae, weight_dtype):
        if not args.network_weights:
            raise ValueError("HiDream objective evaluation requires --network_weights")
        import musubi_tuner

        sys.path.append(str(Path(musubi_tuner.__file__).resolve().parent))
        network_module = importlib.import_module(args.network_module)
        weights_sd = _weights_with_default_alphas(args.network_weights)
        network = network_module.create_arch_network_from_weights(1, weights_sd, unet=transformer)
        if hasattr(network_module, "prepare_network"):
            network.prepare_network(args)
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        network.load_weights(args.network_weights)

        with safe_open(args.network_weights, framework="pt") as handle:
            file_keys = set(handle.keys())
        network_keys = set(network.state_dict().keys())
        unmatched = sorted(file_keys - network_keys)
        if unmatched:
            raise ValueError(f"LoRA does not match HiDream O1: {len(unmatched)} unmatched keys {unmatched[:3]}")
        if not any(
            parameter.detach().abs().max().item() > 0
            for name, parameter in network.named_parameters()
            if "lora_up" in name
        ):
            raise ValueError("LoRA loaded no non-zero up weights; refusing to score the base model")

        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
            network.enable_gradient_checkpointing()
        return network

    def process_batch(self, args, accelerator, transformer, network, batch, latents, noise, *rest):
        if batch["latents"].shape[0] != 1:
            raise ValueError("Objective evaluation requires batch_size=1")
        key = self.sample_keys.get(_fingerprint(batch["latents"][0]))
        if key is None:
            raise ValueError("Batch does not match any held-out sample cache")
        _, repeat = self.seeder.next(key)
        noise = torch.randn_like(latents)
        self.current_key = key
        with torch.no_grad():
            loss, metrics = super().process_batch(args, accelerator, transformer, network, batch, latents, noise, *rest)
        masked = bool(self.mask_dir and (self.mask_dir / f"{key}.png").is_file())
        self.records.append({"path": key, "repeat": repeat, "masked": masked, "loss": float(loss.item())})
        if accelerator.is_main_process:
            self.output.write_text(json.dumps(self.records, allow_nan=False), encoding="utf-8")
        # A leaf with no path to the LoRA keeps backward and the optimizer step no-ops.
        return loss.detach().requires_grad_(), metrics

    def compute_loss(self, args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step):
        mask_path = self.mask_dir / f"{self.current_key}.png" if self.mask_dir else None
        if mask_path is None or not mask_path.is_file():
            return super().compute_loss(args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step)
        if args.weighting_scheme != "none":
            raise ValueError("Masked HiDream objective evaluation requires --weighting_scheme none")
        height_patches, width_patches = output.extra["pixel_grid_hw"]
        height = height_patches * hidream_o1_utils.PATCH_SIZE
        width = width_patches * hidream_o1_utils.PATCH_SIZE
        pred = hidream_o1_utils.unpatchify_pixels(output.pred.float(), height, width)
        target = hidream_o1_utils.unpatchify_pixels(output.target.float(), height, width)
        weights = torch.from_numpy(mask_weights(cover_and_center_crop(Image.open(mask_path), width, height)))
        weights = weights.to(pred.device)[None, None]
        weights = weights / weights.mean()
        return ((pred - target) ** 2 * weights).mean(), {}


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Musubi torch cannot use CUDA; refusing to score on CPU")
    enable_deterministic_kernels()
    parser = hidream_o1_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    GODHiDreamObjectiveEvaluator().train(args)


if __name__ == "__main__":
    main()
