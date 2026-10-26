"""Typed text-eval settings, replacing the axolotl yaml config the evaluators used to carry.

Only the keys the text evaluators actually read survive. The values are the ones that yaml set, and the
sequence_len halving rule is the one the old config loader applied, so a model scores the same held-out rows it
did before. `train_on_inputs` is deliberately absent: prompts are never scored.
"""

from math import ceil

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


DEFAULT_EVAL_SEQUENCE_LEN = 4096


class TextEvalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    sequence_len: int = Field(default=DEFAULT_EVAL_SEQUENCE_LEN, ge=1)
    min_sequence_len: int = Field(default=2, ge=1)
    # Batch size 1 is what makes HF's eval_loss a flat mean of per-example means; find_executable_batch_size
    # only ever shrinks from here.
    starting_batch_size: int = Field(default=1, ge=1)
    log_interval_seconds: int = 10
    output_dir: str = "outputs/"


def text_eval_config_for_model(max_position_embeddings: int | None) -> TextEvalConfig:
    """Halve the eval window for short-context models, exactly as the axolotl-config path did.

    A model whose context is under twice the default window gets ceil(context / 2), so rows near the default
    length are dropped rather than scored against a position range the model barely covers.
    """
    sequence_len = DEFAULT_EVAL_SEQUENCE_LEN
    if max_position_embeddings and max_position_embeddings < 2 * sequence_len:
        sequence_len = ceil(max_position_embeddings / 2)
    return TextEvalConfig(sequence_len=sequence_len)
