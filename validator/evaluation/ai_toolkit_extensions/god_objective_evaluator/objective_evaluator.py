import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class GODObjectiveEvaluator(SDTrainer):
    """Run AI Toolkit's training forward/loss path without updating weights."""

    def __init__(self, process_id, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.evaluation_output = Path(config["evaluation_output"])
        self.evaluation_seed = int(config.get("evaluation_seed", 42000))
        self.evaluation_records = []

    def _seed_case(self) -> None:
        seed = self.evaluation_seed + len(self.evaluation_records)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def hook_train_loop(self, batch):
        batches = batch if isinstance(batch, list) else [batch]
        if len(batches) != 1 or len(batches[0].file_items) != 1:
            raise ValueError("Objective evaluation requires batch_size=1 and no accumulation")

        self._seed_case()
        original_backward = self.accelerator.backward
        self.accelerator.backward = lambda *_args, **_kwargs: None
        try:
            loss = self.train_single_accumulation(batches[0])
        finally:
            self.accelerator.backward = original_backward

        item = batches[0].file_items[0]
        self.evaluation_records.append(
            {"path": str(item.path), "loss": float(loss.item())}
        )
        self.evaluation_output.write_text(
            json.dumps(self.evaluation_records, allow_nan=False), encoding="utf-8"
        )
        self.end_of_training_loop()
        return OrderedDict(loss=float(loss.item()))

    def save(self, step=None):
        # Evaluation must never mutate or emit a candidate checkpoint.
        return None
