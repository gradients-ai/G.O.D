from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from core.models.payload_models import PreparedTrainingImage


class ImageBuildVariant(str, Enum):
    TEXT = "text"
    IMAGE_FLUX = "image-flux"
    IMAGE_TOOLKIT = "image-toolkit"


class ImageBuildStatus(str, Enum):
    QUEUED = "queued"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class ImageBuild(BaseModel):
    build_id: str
    tournament_id: str
    hotkey: str
    variant: ImageBuildVariant
    repo: str
    commit_ref: str
    repository: str
    status: ImageBuildStatus
    attempts: int
    next_attempt_at: datetime
    builder: str | None = None
    lease_expires_at: datetime | None = None
    digest: str | None = None
    resolved_commit: str | None = None
    failure_class: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    @property
    def tag(self) -> str:
        return f"b{self.build_id.replace('-', '')[:16]}-a{self.attempts}"

    def prepared_image(self) -> PreparedTrainingImage | None:
        if self.status != ImageBuildStatus.READY or not self.digest:
            return None
        return PreparedTrainingImage(
            build_id=self.build_id, repository=self.repository, digest=self.digest, variant=self.variant.value
        )
