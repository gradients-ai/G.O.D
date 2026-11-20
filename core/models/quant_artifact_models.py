from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from core.constants.quant import QUANT_MANIFEST_MAX_FILES
from core.constants.quant import QUANT_MANIFEST_NAME
from core.models.task_models import TaskType


QuantTrack = Literal["fixed", "freeform"]
_TRACKS: dict[TaskType, QuantTrack] = {TaskType.QUANTFIXEDTASK: "fixed", TaskType.QUANTFREEFORMTASK: "freeform"}


def track_for(task_type: TaskType) -> QuantTrack:
    """The manifest track of a quant task type. Raises KeyError for any other type."""
    return _TRACKS[task_type]


class QuantManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Path relative to the output dir root, '/'-separated")
    bytes: int = Field(..., ge=0)

    @field_validator("name")
    @classmethod
    def validate_relative_name(cls, v: str) -> str:
        path = PurePosixPath(v)
        if not v or v.startswith("/") or path.is_absolute():
            raise ValueError(f"file name {v!r} must be relative")
        if ".." in path.parts:
            raise ValueError(f"file name {v!r} must not contain '..'")
        if path == PurePosixPath(QUANT_MANIFEST_NAME):
            raise ValueError(f"file name must not be {QUANT_MANIFEST_NAME}")
        return v


class QuantManifest(BaseModel):
    """quant_manifest.json, written last by the miner's pipeline: the complete list of files it declares as its output."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    task_id: str
    track: QuantTrack
    files: list[QuantManifestFile] = Field(..., min_length=1, max_length=QUANT_MANIFEST_MAX_FILES)

    @field_validator("files")
    @classmethod
    def validate_unique_names(cls, v: list[QuantManifestFile]) -> list[QuantManifestFile]:
        names = [PurePosixPath(f.name) for f in v]
        if len(set(names)) != len(names):
            raise ValueError("file names must be unique")
        return v


class QuantDecodeRequest(BaseModel):
    """What the validator hands the trainer to decode a freeform submission after its HF upload."""

    model_config = ConfigDict(extra="forbid")

    decoder_snapshot_url: str = Field(..., description="Presigned GET of the decoder snapshot .tar.xz in MinIO")
    decoder_sha256: str = Field(..., description="xz_sha256 of the decoder snapshot")
    bucket: str = Field(..., description="MinIO bucket the decoded bf16 shards are uploaded to")
    object_prefix: str = Field(..., description="Validator-chosen object prefix, e.g. quant-decoded/<task_id>/<hotkey>")


class QuantDecodedFile(BaseModel):
    name: str
    bytes: int
    sha256: str


class QuantDecodeResult(BaseModel):
    """Outcome of one trainer-side freeform decode. Exactly one of ok, quant_failure, infra_error is set.

    quant_failure is the miner's fault and is a scored fail; infra_error is ours (image missing,
    MinIO/HF unreachable, snapshot sha mismatch) and must never be scored.
    """

    ok: bool = False
    quant_failure: str | None = None
    infra_error: str | None = None
    artifact_bytes: int | None = None
    decoder_bytes: int | None = None
    decoder_sha256: str | None = None
    n_tensors: int = 0
    uploaded_files: list[str] = []
    build_seconds: float = 0.0
    run_seconds: float = 0.0
    index_object: str | None = None
    # Set when build.sh exited 0 but left /build empty or logged a failure line: the decoder likely ran its
    # fallback. Never a scored fail (falling back is the miner's choice); persisted to task_nodes.decoder_build_warning.
    build_warning: str | None = None
    # Every regular file the decode downloaded and counted in artifact_bytes (relative posix name, .gitattributes
    # included, sorted by name), and the checked bf16 shards it uploaded.
    artifact_files: list[QuantDecodedFile] = []
    decoded_files: list[QuantDecodedFile] = []

    @model_validator(mode="after")
    def validate_single_outcome(self) -> "QuantDecodeResult":
        outcomes = [self.ok, self.quant_failure is not None, self.infra_error is not None]
        if sum(outcomes) != 1:
            raise ValueError("exactly one of ok, quant_failure, infra_error must be set")
        return self


class QuantDecodedIndex(BaseModel):
    """index.json under the decode object prefix. Uploaded last: its presence marks the decoded upload complete."""

    files: list[QuantDecodedFile]
    n_tensors: int
    decoder_sha256: str
    artifact_bytes: int
    task_id: str
    hotkey: str
    artifact_files: list[QuantDecodedFile] = []
