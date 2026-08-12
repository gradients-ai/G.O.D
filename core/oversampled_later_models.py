import json
import random
from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict


_POOL_PATH = Path(__file__).parent / "oversampled_later_models.json"


class OversampledLaterModel(BaseModel):
    """A 2026+ base model the text tournament oversamples into a guaranteed task slot.

    params_b is the safetensors parameter count at the time of listing — informational only,
    task sizing still reads the live count from the hub (get_model_num_params).
    """

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    params_b: float
    model_type: str


class OversampledLaterModelPool(BaseModel):
    comment: str
    models: list[OversampledLaterModel]


def _load() -> OversampledLaterModelPool:
    raw = json.loads(_POOL_PATH.read_text())
    return OversampledLaterModelPool(comment=raw["_comment"], models=raw["models"])


# Ordered pool, sampled uniformly (see sample_oversampled_later_model). Kept in-repo until the
# content service can serve models by release date — see the JSON's _comment.
OVERSAMPLED_LATER_MODELS: list[OversampledLaterModel] = _load().models


def sample_oversampled_later_model(rng: random.Random | None = None) -> str:
    """Pick one model id from the pool, uniformly.

    Pass a seeded rng where the choice must survive a retry — tournament task creation is
    resumable, so re-running a partially created round has to land on the same model.
    """
    return (rng or random).choice(OVERSAMPLED_LATER_MODELS).model_id
