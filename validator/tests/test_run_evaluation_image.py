import asyncio

from core.models.utility_models import ImageModelType
from validator.evaluation.docker_evaluation import run_evaluation_docker_image
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def test():
    test_split_url = "https://gradients.s3.eu-north-1.amazonaws.com/hamzatest.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20251216%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251216T172018Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=a56135c96ae3e9f8200b9cc102a5c1852857d5337b0cf47cc7c9af288b8b405b"
    
    results = await run_evaluation_docker_image(
        test_split_url=test_split_url,
        original_model_repo="Qwen/Qwen-Image",
        models=['gradients-io-tournaments/qwenimage-test'],
        model_type=ImageModelType.QWEN_IMAGE,
        gpu_ids=[0]
    )
    logger.info(f"Evaluation results: {results}")


if __name__ == "__main__":
    asyncio.run(test())