from datetime import UTC
from datetime import datetime
from uuid import uuid4

from core.models.image_models import ImageModelType
from core.models.task_models import TaskStatus
from validator.tasks.models import ImageRawTask
from validator.tasks.requests import prepare_image_task_request


def test_prepare_image_task_request_propagates_trigger_word():
    task = ImageRawTask(
        is_organic=False,
        status=TaskStatus.PENDING,
        model_id="black-forest-labs/FLUX.1-dev",
        ds="image-dataset",
        account_id=uuid4(),
        hours_to_complete=3,
        training_data="s3://bucket/images.zip",
        created_at=datetime.now(UTC),
        model_type=ImageModelType.FLUX,
        trigger_word="glimmerforge",
    )

    request = prepare_image_task_request(task)

    assert request.trigger_word == "glimmerforge"
    assert request.dataset_zip == "s3://bucket/images.zip"
    assert request.model_type == ImageModelType.FLUX
