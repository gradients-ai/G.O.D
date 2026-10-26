import pytest
from pydantic import ValidationError

from validator.evaluation.text_eval_config import DEFAULT_EVAL_SEQUENCE_LEN
from validator.evaluation.text_eval_config import TextEvalConfig
from validator.evaluation.text_eval_config import text_eval_config_for_model


def test_defaults_match_the_old_eval_yaml():
    config = TextEvalConfig()
    assert config.sequence_len == DEFAULT_EVAL_SEQUENCE_LEN == 4096
    assert config.min_sequence_len == 2
    assert config.starting_batch_size == 1
    assert config.log_interval_seconds == 10
    assert config.output_dir == "outputs/"


@pytest.mark.parametrize(
    "max_position_embeddings, expected",
    [
        (40960, 4096),
        (8192, 4096),  # not strictly below 2 * 4096
        (8191, 4096),  # ceil(8191 / 2)
        (4000, 2000),
        (4097, 2049),
        (512, 256),
        (None, 4096),
        (0, 4096),
    ],
)
def test_halving_rule(max_position_embeddings, expected):
    assert text_eval_config_for_model(max_position_embeddings).sequence_len == expected


def test_config_is_frozen_and_copied_for_the_retry_ladder():
    config = TextEvalConfig()
    with pytest.raises(ValidationError):
        config.sequence_len = 8192
    assert config.model_copy(update={"sequence_len": 8192}).sequence_len == 8192
    assert config.sequence_len == 4096


def test_rejects_nonpositive_sequence_len():
    with pytest.raises(ValidationError):
        TextEvalConfig(sequence_len=0)
