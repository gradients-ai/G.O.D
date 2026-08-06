import pytest

from validator.tasks import requests


def _fail_model_info(model_id):
    raise RuntimeError(f"no metadata for {model_id}")


@pytest.mark.parametrize(
    ("model_id", "expected_params"),
    [
        ("org/custom-0.8B", 800_000_000),
        ("org/custom-2.6B", 2_600_000_000),
        ("org/custom-7B-Instruct", 7_000_000_000),
    ],
)
def test_model_size_regex_fallback_preserves_decimal_sizes(monkeypatch, model_id, expected_params):
    monkeypatch.setattr(requests.hf_api, "model_info", _fail_model_info)

    assert requests.get_model_num_params(model_id) == expected_params


@pytest.mark.parametrize(
    ("model_id", "expected_params"),
    [
        ("Qwen/Qwen3.5-4B", 5_000_000_000),
        ("google/gemma-4-E2B", 5_100_000_000),
    ],
)
def test_known_tournament_model_size_metadata_precedes_name_fallback(monkeypatch, model_id, expected_params):
    monkeypatch.setattr(requests.hf_api, "model_info", _fail_model_info)

    assert requests.get_model_num_params(model_id) == expected_params
