# This script is meant to simplify recreating the evaluation process for a specific task.
# It runs the same evaluation docker as the validator does. Make sure to build the latest docker images before running this script
# docker build -f dockerfiles/validator.dockerfile -t weightswandering/tuning_vali:latest .
# docker build -f dockerfiles/validator-diffusion.dockerfile -t diagonalge/tuning_validator_diffusion:latest .

import argparse
import asyncio
import json
import os
from typing import List
from typing import Optional

import httpx

from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import TaskType
from core.utils import download_s3_file
from validator.core.models import ChatTaskWithHotkeyDetails
from validator.core.models import DpoTaskWithHotkeyDetails
from validator.core.models import GrpoTaskWithHotkeyDetails
from validator.core.models import ImageTaskWithHotkeyDetails
from validator.core.models import EnvTaskWithHotkeyDetails
from validator.core.models import InstructTextTaskWithHotkeyDetails
from validator.evaluation.docker_evaluation import run_evaluation_docker_image
from validator.evaluation.docker_evaluation import run_evaluation_docker_text
from validator.utils.logging import get_logger


logger = get_logger(__name__)

API_BASE_URL = "https://api.gradients.io"


async def fetch_task_details(task_id: str):
    """
    Fetch task details from the Gradients API.

    Args:
        task_id: The ID of the task to fetch

    Returns:
        Task details as the appropriate task type with hotkey details
    """
    url = f"{API_BASE_URL}/auditing/tasks/{task_id}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch task details: {response.status_code} {response.text}")

        data = response.json()
        task_type = data.get("task_type")

        if task_type == TaskType.INSTRUCTTEXTTASK.value:
            return InstructTextTaskWithHotkeyDetails(**data)
        elif task_type == TaskType.CHATTASK.value:
            return ChatTaskWithHotkeyDetails(**data)
        elif task_type == TaskType.IMAGETASK.value:
            return ImageTaskWithHotkeyDetails(**data)
        elif task_type == TaskType.DPOTASK.value:
            return DpoTaskWithHotkeyDetails(**data)
        elif task_type == TaskType.GRPOTASK.value:
            return GrpoTaskWithHotkeyDetails(**data)
        elif task_type == TaskType.ENVIRONMENTTASK.value:
            return EnvTaskWithHotkeyDetails(**data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")


async def run_evaluation_from_task_id(
    task_id: str,
    gpu_ids: List[int] = [0],
    models: Optional[List[str]] = None,
):
    """
    Run model evaluation based on task ID and log the results

    Args:
        task_id: The ID of the task to evaluate
        gpu_ids: List of GPU IDs to use for evaluation
        models: Optional list of specific models to evaluate instead of using hotkey details
    """
    task_details = await fetch_task_details(task_id)
    logger.info(f"Retrieved task details for task {task_id}")

    task_type = task_details.task_type

    original_model = task_details.model_id
    if not original_model:
        raise ValueError("Original model not found in task details")

    test_data_url = task_details.test_data
    if not test_data_url:
        raise ValueError("Test data URL not found in task details")

    if models:
        models_to_evaluate = models
    else:
        models_to_evaluate = []
        if task_details.hotkey_details:
            for hotkey_detail in task_details.hotkey_details:
                if hotkey_detail.repo:
                    models_to_evaluate.append(hotkey_detail.repo)

    if task_type == TaskType.IMAGETASK:
        logger.info(f"Running image evaluation for models: {models_to_evaluate}")
        try:
            results = await run_evaluation_docker_image(
                test_split_url=test_data_url,
                original_model_repo=original_model,
                models=models_to_evaluate,
                model_type=task_details.model_type,
                gpu_ids=gpu_ids,
            )

            results_dict = results.model_dump()
            if "results" in results_dict:
                for model, result in results_dict["results"].items():
                    if isinstance(result, Exception):
                        results_dict["results"][model] = f"ERROR: {str(result)}"
                        logger.error(f"Evaluation failed for model {model}: {result}")

            logger.info(f"Image evaluation results: {json.dumps(results_dict, indent=2)}")
        except Exception as e:
            logger.error(f"Image evaluation failed with error: {e}", exc_info=True)

        return

    elif task_type == TaskType.INSTRUCTTEXTTASK:
        dataset_type = InstructTextDatasetType(
            field_instruction=task_details.field_instruction,
            field_input=task_details.field_input,
            field_output=task_details.field_output,
            field_system=task_details.field_system,
            format=task_details.format,
            no_input_format=task_details.no_input_format,
        )
    elif task_type == TaskType.CHATTASK:
        dataset_type = ChatTemplateDatasetType(
            chat_template=task_details.chat_template,
            chat_column=task_details.chat_column,
            chat_role_field=task_details.chat_role_field,
            chat_content_field=task_details.chat_content_field,
            chat_user_reference=task_details.chat_user_reference,
            chat_assistant_reference=task_details.chat_assistant_reference,
        )
    elif task_type == TaskType.DPOTASK:
        dataset_type = DpoDatasetType(
            field_prompt=task_details.field_prompt,
            field_system=task_details.field_system,
            field_chosen=task_details.field_chosen,
            field_rejected=task_details.field_rejected,
            prompt_format=task_details.prompt_format,
            chosen_format=task_details.chosen_format,
            rejected_format=task_details.rejected_format,
        )
    elif task_type == TaskType.GRPOTASK:
        dataset_type = GrpoDatasetType(field_prompt=task_details.field_prompt, reward_functions=task_details.reward_functions)
    elif task_type == TaskType.ENVIRONMENTTASK:
        dataset_type = EnvironmentDatasetType(environment_name=task_details.environment_name, eval_seed=task_details.eval_seed)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    logger.info("Downloading test and synth data...")
    test_data_path = await download_s3_file(test_data_url)
    synth_data_path = await download_s3_file(task_details.synthetic_data)
    logger.info(f"Downloaded test and synth data to {test_data_path} and {synth_data_path}")

    try:
        logger.info(f"Running test data evaluation for models: {models_to_evaluate}")
        test_data_results = await run_evaluation_docker_text(
            dataset=test_data_path,
            models=models_to_evaluate,
            original_model=original_model,
            dataset_type=dataset_type,
            file_format=FileFormat.JSON,
            gpu_ids=gpu_ids,
        )

        test_data_results_dict = test_data_results.model_dump()
        if "results" in test_data_results_dict:
            for model, result in test_data_results_dict["results"].items():
                if isinstance(result, Exception):
                    test_data_results_dict["results"][model] = f"ERROR: {str(result)}"
                    logger.error(f"Test evaluation failed for model {model}: {result}")

        logger.info(f"Test data evaluation results: {json.dumps(test_data_results_dict, indent=2)}")

        logger.info(f"Running synthetic data evaluation for models: {models_to_evaluate}")
        synth_data_results = await run_evaluation_docker_text(
            dataset=synth_data_path,
            models=models_to_evaluate,
            original_model=original_model,
            dataset_type=dataset_type,
            file_format=FileFormat.JSON,
            gpu_ids=gpu_ids,
        )

        synth_data_results_dict = synth_data_results.model_dump()
        if "results" in synth_data_results_dict:
            for model, result in synth_data_results_dict["results"].items():
                if isinstance(result, Exception):
                    synth_data_results_dict["results"][model] = f"ERROR: {str(result)}"
                    logger.error(f"Synthetic evaluation failed for model {model}: {result}")

        logger.info(f"Synthetic data evaluation results: {json.dumps(synth_data_results_dict, indent=2)}")

    except Exception as e:
        logger.error(f"Text evaluation failed with error: {e}", exc_info=True)
    finally:
        os.remove(test_data_path)
        os.remove(synth_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Run model evaluation for a specific task. 
        The task needs to be not older than 7 days. 
        If older than 7 days the test and synth urls are not valid anymore.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation for a task using the default GPU (0)
  python utils/run_evaluation.py --task_id task_12345
  
  # Run evaluation using multiple GPUs
  python utils/run_evaluation.py --task_id task_12345 --gpu_ids 0 1 2
  
  # Evaluate specific models instead of those in task details
  python utils/run_evaluation.py --task_id task_12345 --models huggingface/model1 huggingface/model2
        """,
    )
    parser.add_argument("--task_id", type=str, required=True, help="Task ID to fetch details from the Gradients API")
    parser.add_argument(
        "--gpu_ids", nargs="+", type=int, default=[0], help="List of GPU IDs to use for evaluation (default: [0])"
    )
    parser.add_argument(
        "--models", nargs="+", help="Optional list of specific models to evaluate instead of using models from task details"
    )

    args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if v is not None}

    asyncio.run(run_evaluation_from_task_id(**kwargs))
