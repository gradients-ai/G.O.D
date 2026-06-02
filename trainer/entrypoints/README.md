# Trainer Entrypoints

Scripts executed inside miner training images.

## Contents

- `text_trainer.py`: reference text-family trainer for Instruct, Chat, DPO, GRPO, and environment tasks.
- `image_trainer.py`: reference image/diffusion trainer.
- `__init__.py`: package marker.

Miner repositories may replace these implementations, but their containers must still obey the CLI, environment, and output contracts described in `docs/miner.md`.
