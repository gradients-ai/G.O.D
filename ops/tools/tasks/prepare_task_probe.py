import asyncio

from core.logging import get_logger
from core.logging import redact_url_queries
from validator.app.config import load_config
from validator.tasks.datasets.preparation import prepare_text_task
from validator.tasks.models import ChatRawTask


logger = get_logger(__name__)


async def main():
    task = ChatRawTask(
        chat_template="chatml",
        chat_assistant_reference="gpt",
        chat_user_reference="human",
        chat_content_field="value",
        chat_role_field="from",
        hours_to_complete=1,
        account_id="3fa85f64-5717-4562-b3fc-2c963f66afa6",
        ds_repo="Magpie-Align/Magpie-Pro-300K-Filtered",
        file_format="hf",
        is_organic=True,
        status="pending",
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        ds="Magpie-Align/Magpie-Pro-300K-Filtered",
        created_at="2023-10-01T00:00:00Z",
    )

    config = load_config()

    try:
        split = await prepare_text_task(task=task, keypair=config.keypair)

        logger.info(redact_url_queries(f"Test data URL: {split.test_url}"))
        logger.info(redact_url_queries(f"Training data URL: {split.train_url}"))

    except Exception as e:
        logger.error(f"Error in task preparation: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
