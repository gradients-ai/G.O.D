# SFT dataset generator

Plays game episodes against an environment server with an LLM player and writes them as an instruct-style dataset.

## Contents

| Path | Purpose |
| --- | --- |
| `generate.py` | Main script: runs episodes per game and writes the dataset. |
| `config.py` | Game configuration and argument parsing. |
| `claude_client.py` | LLM player client with token usage tracking. |
| `env_client.py` | Environment server client. |
| `game_prompts.py` | Prompt templates per game. |
| `dataset_formatter.py` | Checkpointing, formatting and validation of recorded episodes. |
| `requirements.txt` | Python requirements of the tool. |
| `run.sh` | Shell wrapper that runs generation. |
