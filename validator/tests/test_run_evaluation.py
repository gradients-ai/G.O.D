import asyncio

from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import FileFormat
from validator.evaluation.docker_evaluation import run_evaluation_docker_text
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def test():
    custom_dataset_type = ChatTemplateDatasetType(
        chat_template='chatml',
        chat_column='conversations',
        chat_role_field='from',
        chat_content_field='value',
        chat_user_reference='human',
        chat_assistant_reference='gpt'
    )

    results = await run_evaluation_docker_text(
        dataset="/tmp/728c0ffac41d1699_test_data.json",
        models=['diagonalge/8ad2b90f-7b3e-4b67-9741-3f3c2ecc53eb'],
        original_model="Qwen/Qwen2.5-0.5B",
        dataset_type=custom_dataset_type,
        file_format=FileFormat.JSON,
        gpu_ids=[0]
    )
    logger.info(f"Evaluation results: {results}")


if __name__ == "__main__":
    asyncio.run(test())