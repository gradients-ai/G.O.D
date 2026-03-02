#!/usr/bin/env python3
"""Run Basilica text evaluation. Edit the params below and run: python -m scripts.manual_basilica_text_eval"""

import asyncio
import os

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".vali.env"))

from core.models.utility_models import GrpoDatasetType, RewardFunction
from core.models.utility_models import FileFormat
from validator.evaluation.docker_evaluation import run_evaluation_basilica_text

# --- Edit these ---
DATASET = "https://s3.eu-central-003.backblazeb2.com/gradients-validator/364be2bd9364f21c_test_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=00362e8d6b742200000000002%2F20260228%2Feu-central-003%2Fs3%2Faws4_request&X-Amz-Date=20260228T100934Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=092e38563ece490f4bf49791c2003217fd210f2eb223b0917a82bb2cac465db2"
MODELS = ["gradients-io-tournaments/tournament-tourn_0cb47dcc90741bd7_20260226-c6ada604-7c4d-40d0-a648-8d476d04f631-5Ckjsrw4", "gradients-io-tournaments/tournament-tourn_0cb47dcc90741bd7_20260226-c6ada604-7c4d-40d0-a648-8d476d04f631-5GU4Xkd3"]
ORIGINAL_MODEL = "stabilityai/japanese-stablelm-instruct-beta-70b"
DATASET_TYPE = GrpoDatasetType(
    field_prompt="prompt",
    reward_functions=[
        RewardFunction(
            reward_id="02cf0da9-9172-4f70-a97a-e35782126713",
            reward_func=(
                'def reward_think_answer_format(completions, **kwargs):\n'
                '    """Reward function that checks if the completion has a specific format."""\n'
                '    import re\n'
                '    pattern = r"^<think>.*?</think><answer>.*?</answer>$"\n'
                '    try:\n'
                '        matches = [re.match(pattern, content) for content in completions]\n'
                '        return [1.0 if match else 0.0 for match in matches]\n'
                '    except Exception as e:\n'
                '        print(f"Error in format_reward_func: {e}")\n'
                '        return [0.0 for _ in completions]\n'
            ),
            reward_weight=0.5,
            func_hash="e5bab80edb8fe0a76e0018c15df56e3e52f2c932e8506c6f7cf61ba59c10defa",
            is_generic=True,
            is_manual=True,
        ),
        RewardFunction(
            reward_id="a5acebbb-d649-49c5-9dfd-83465eb0259f",
            reward_func=(
                'def reward_specific_char_count(completions, **kwargs):\n'
                '    """Rewards completions that are close to n_chars characters."""\n'
                '    n_chars = 100\n'
                '    return [-abs(n_chars - len(completion)) for completion in completions]\n'
            ),
            reward_weight=0.5,
            func_hash="db0451506ebb04b0f56e4295b1381f499c3634f25d3cd3002b041ab10dbac24f",
            is_generic=True,
            is_manual=True,
        ),
    ],
)
FILE_FORMAT = FileFormat.HF
NUM_GPUS = 4
EVAL_SEED = 561987444
# ---

if __name__ == "__main__":
    asyncio.run(
        run_evaluation_basilica_text(
            dataset=DATASET,
            models=MODELS,
            original_model=ORIGINAL_MODEL,
            dataset_type=DATASET_TYPE,
            file_format=FILE_FORMAT,
            num_gpus=NUM_GPUS,
        )
    )
