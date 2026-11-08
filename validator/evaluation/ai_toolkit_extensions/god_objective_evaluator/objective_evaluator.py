import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from PIL import Image

from validator.evaluation.edit_masks import mask_as_float
from validator.evaluation.edit_masks import mask_weights
from validator.evaluation.edit_masks import region_mean_losses
from validator.evaluation.edit_masks import resize_and_crop
from validator.evaluation.objective_seeding import CaseSeeder
from validator.evaluation.objective_seeding import case_seed
from validator.evaluation.objective_seeding import enable_deterministic_kernels
from validator.evaluation.objective_seeding import seed_everything


class GODObjectiveEvaluator(SDTrainer):
    """Run AI Toolkit's training forward/loss path without updating weights."""

    def __init__(self, process_id, job, config: OrderedDict, **kwargs):
        enable_deterministic_kernels()
        super().__init__(process_id, job, config, **kwargs)
        self.evaluation_output = Path(config["evaluation_output"])
        self.evaluation_seeder = CaseSeeder(int(config.get("evaluation_seed", 42000)))
        mask_dir = config.get("evaluation_mask_dir")
        self.evaluation_mask_dir = Path(mask_dir) if mask_dir else None
        self.evaluation_records = []
        self.candidate_lora_loaded = False
        self.mask_shape_checked = False
        self._last_region_losses: dict | None = None

    def hook_after_sd_init_before_load(self):
        super().hook_after_sd_init_before_load()
        encode_images = self.sd.encode_images
        base_seed = self.evaluation_seeder.base_seed

        def seeded_encode_images(image_list, *args, **kwargs):
            # Some VAEs sample their posterior on the global RNG while the latent cache is
            # built, which differs per container; seed each encode from the image content
            # and restore the caller's RNG so per-case noise is untouched.
            images = image_list if isinstance(image_list, (list, tuple)) else [image_list]
            digest = hashlib.sha256()
            for image in images:
                digest.update(image.detach().float().cpu().contiguous().numpy().tobytes())
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                seed_everything(case_seed(base_seed, digest.hexdigest(), 0))
                return encode_images(image_list, *args, **kwargs)

        self.sd.encode_images = seeded_encode_images

    def load_weights(self, path):
        extra_weights = super().load_weights(path)
        if extra_weights:
            examples = ", ".join(sorted(extra_weights)[:3])
            raise ValueError(f"{len(extra_weights)} LoRA keys match no evaluation module (e.g. {examples})")
        if not any(
            parameter.detach().abs().max().item() > 0
            for name, parameter in self.network.named_parameters()
            if "lora_up" in name
        ):
            raise ValueError("LoRA loaded no non-zero up weights; refusing to score the base model")
        self.candidate_lora_loaded = True
        return extra_weights

    def _loss_target(self, noise_pred, noise, noisy_latents, timesteps, batch, prior_pred=None):
        """Mirror SDTrainer's primary target selection for region metrics."""
        if hasattr(self.sd, "get_loss_target"):
            return self.sd.get_loss_target(noise=noise, batch=batch, timesteps=timesteps).detach()
        if self.sd.is_flow_matching:
            return (noise - batch.latents).detach()
        if self.sd.prediction_type == "v_prediction":
            return self.sd.noise_scheduler.get_velocity(batch.tensor, noise, timesteps)
        if prior_pred is not None and not self.train_config.do_prior_divergence:
            return prior_pred
        return noise

    def calculate_loss(self, noise_pred, noise, noisy_latents, timesteps, batch, mask_multiplier=1.0, prior_pred=None, **kwargs):
        self._last_region_losses = None
        if batch.mask_tensor is not None:
            if not isinstance(mask_multiplier, torch.Tensor) or mask_multiplier.shape[-2:] != noise_pred.shape[-2:]:
                raise ValueError("Edit mask did not reach the loss at the prediction's spatial size")
            torch.broadcast_shapes(mask_multiplier.shape, noise_pred.shape)
            self.mask_shape_checked = True
            mask_01 = getattr(batch, "_god_edit_mask_01", None)
            if mask_01 is not None and not isinstance(noise_pred, dict):
                target = self._loss_target(noise_pred, noise, noisy_latents, timesteps, batch, prior_pred)
                per_pixel = torch.nn.functional.mse_loss(
                    noise_pred.float(), target.float(), reduction="none"
                )
                changed, unchanged, coverage = region_mean_losses(per_pixel, mask_01)
                self._last_region_losses = {
                    "loss_changed": changed,
                    "loss_unchanged": unchanged,
                    "mask_coverage": coverage,
                }
        return super().calculate_loss(
            noise_pred=noise_pred,
            noise=noise,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            batch=batch,
            mask_multiplier=mask_multiplier,
            prior_pred=prior_pred,
            **kwargs,
        )

    def _attach_edit_mask(self, batch, item) -> bool:
        if self.evaluation_mask_dir is None:
            return False
        mask_path = self.evaluation_mask_dir / f"{Path(item.path).stem}.png"
        if not mask_path.is_file():
            return False
        if item.flip_x or item.flip_y:
            raise ValueError("Objective evaluation masks do not support flipped samples")
        mask = resize_and_crop(
            Image.open(mask_path),
            (item.scale_to_width, item.scale_to_height),
            (item.crop_x, item.crop_y, item.crop_x + item.crop_width, item.crop_y + item.crop_height),
        )
        values = mask_as_float(mask)
        batch._god_edit_mask_01 = torch.from_numpy(values).unsqueeze(0).unsqueeze(0)
        weights = torch.from_numpy(mask_weights(mask))
        batch.mask_tensor = weights.unsqueeze(0).unsqueeze(0)
        return True

    def hook_train_loop(self, batch):
        batches = batch if isinstance(batch, list) else [batch]
        if len(batches) != 1 or len(batches[0].file_items) != 1:
            raise ValueError("Objective evaluation requires batch_size=1 and no accumulation")
        if not self.candidate_lora_loaded:
            raise ValueError("Candidate LoRA was not loaded; refusing to score the base model")

        item = batches[0].file_items[0]
        sample_key = Path(item.path).stem
        masked = self._attach_edit_mask(batches[0], item)
        self.mask_shape_checked = False
        self._last_region_losses = None
        _, repeat = self.evaluation_seeder.next(sample_key)
        original_backward = self.accelerator.backward
        self.accelerator.backward = lambda *_args, **_kwargs: None
        try:
            loss = self.train_single_accumulation(batches[0])
        finally:
            self.accelerator.backward = original_backward
        if masked and not self.mask_shape_checked:
            raise ValueError("Edit mask was attached but never reached the loss")

        # SDTrainer replaces non-finite training losses with zeros before return. Treat a
        # zero/non-finite case loss as evaluation failure rather than a perfect score.
        case_loss = float(loss.item())
        if not torch.isfinite(loss).item() or case_loss == 0.0:
            raise ValueError(f"Non-finite or zero objective loss for {sample_key}: {case_loss}")

        record = {"path": str(item.path), "repeat": repeat, "masked": masked, "loss": case_loss}
        if self._last_region_losses:
            record.update(self._last_region_losses)
        self.evaluation_records.append(record)
        self.evaluation_output.write_text(
            json.dumps(self.evaluation_records, allow_nan=False), encoding="utf-8"
        )
        self.end_of_training_loop()
        return OrderedDict(loss=case_loss)

    def save(self, step=None):
        # Evaluation must never mutate or emit a candidate checkpoint.
        return None
