<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/banner-dark.svg"><img src="docs/assets/readme/banner-light.svg" alt="G.O.D: Gradients on Demand, Bittensor subnet 56" width="100%"></picture>
</p>

<h3 align="center">The best training recipes are found by open competition.</h3>

<p align="center">
  Every week, miners' code trains text and image models and compresses language models on the same tasks and the same hardware. Up to three places per tournament are paid from the subnet's miner emissions, and the winning code is <a href="https://github.com/gradients-opensource">published in the open</a>.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2506.07940"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/cta-paper-dark.svg"><img src="docs/assets/readme/cta-paper-light.svg" alt="Read the paper" height="40"></picture></a>
  <a href="https://gradients.io"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/cta-train-dark.svg"><img src="docs/assets/readme/cta-train-light.svg" alt="Train on gradients.io" height="40"></picture></a>
  <a href="docs/miner.md"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/cta-compete-dark.svg"><img src="docs/assets/readme/cta-compete-light.svg" alt="Compete as a miner" height="40"></picture></a>
</p>

<br>

<p align="center">
  <a href="https://arxiv.org/abs/2506.07940"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/stat-1-dark.svg"><img src="docs/assets/readme/stat-1-light.svg" alt="100% win rate against TogetherAI, Databricks and Google Cloud" width="270"></picture></a>
  <a href="https://arxiv.org/abs/2506.07940"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/stat-2-dark.svg"><img src="docs/assets/readme/stat-2-light.svg" alt="82.8% of experiments won against HuggingFace AutoTrain" width="270"></picture></a>
  <a href="https://arxiv.org/abs/2506.07940"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/stat-3-dark.svg"><img src="docs/assets/readme/stat-3-light.svg" alt="180 fine-tuning tasks, on models from 70M to 70B parameters" width="270"></picture></a>
  <br>
  Results reported in <a href="https://arxiv.org/abs/2506.07940">Gradients: When Markets Meet Fine-tuning</a> (arXiv:2506.07940).
</p>

## Three tournaments

<p align="center">
  <a href="docs/miner_text.md"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/card-text-dark.svg"><img src="docs/assets/readme/card-text-light.svg" alt="Text tournament. Fine-tune a given language model on an instruct, chat, DPO or GRPO task. Ranked by lowest loss on held-out data; on GRPO, highest reward. Open the text guide." width="270"></picture></a>
  <a href="docs/miner_image.md"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/card-image-dark.svg"><img src="docs/assets/readme/card-image-light.svg" alt="Image tournament. Train a LoRA for a given image model on a text-to-image or edit task. Ranked by lowest loss on held-out images. Open the image guide." width="270"></picture></a>
  <a href="docs/miner_compression.md"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/card-compression-dark.svg"><img src="docs/assets/readme/card-compression-light.svg" alt="Compression tournament. Compress a given language model: a fixed quantised format, or freeform, any format with your own decoder. Ranked by lowest loss in the fixed format; in freeform by fewest bytes, with a penalty on loss. Open the compression guide." width="270"></picture></a>
</p>

## How it works

Miners compete with training code. Each miner enters a tournament with a GitHub repository at a pinned commit. For every task the validator runs that code on its own GPUs, then evaluates and ranks the resulting models.

<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/steps-1-2-dark.svg"><img src="docs/assets/readme/steps-1-2-light.svg" alt="01 Enter: a GitHub repository at a pinned commit. 02 Rounds: same tasks, same hardware; one challenger remains." width="405"></picture>
  <picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/steps-3-4-dark.svg"><img src="docs/assets/readme/steps-3-4-light.svg" alt="03 Boss round: the challenger is compared with the boss, task by task. 04 Crown: beat the boss to take it; the winning code is published." width="405"></picture>
</p>

The crown is decided head-to-head against the boss, the defending champion. That is seven tasks in every tournament: the three round tasks, which the boss also runs, and the four boss-round tasks. The challenger needs at least five wins, under the [crown rule](docs/miner.md#the-boss-and-the-crown); otherwise the boss keeps the crown.

## Start here

Miners start at the [miner hub](docs/miner.md).

## Documentation

| I want to... | Read |
| --- | --- |
| Enter a tournament | [Miner hub](docs/miner.md) |
| Know one tournament's rules | [Text](docs/miner_text.md), [image](docs/miner_image.md) or [compression](docs/miner_compression.md) guide |
| Know how the crown is decided | [The boss and the crown](docs/miner.md#the-boss-and-the-crown) |
| Know how emissions are paid | [Payouts](docs/miner.md#payouts) |
| See results and balances | [Results](docs/miner.md#results) |
| Run a validator or a trainer, or change this codebase | [Developer guide](docs/developer.md) |
| Use the Gradients API | [SKILL.md](SKILL.md) |

## Licence

Apache 2.0. See [LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
