"""get_model_num_params bounds its Hub lookup and falls back to the name when the Hub stalls."""

import requests

import validator.tasks.datasets.constants as cst
from validator.tasks import requests as task_requests


def test_model_info_is_called_with_timeout(monkeypatch):
    seen = {}

    def fake_model_info(model_id, timeout=None):
        seen["timeout"] = timeout
        raise requests.exceptions.ReadTimeout("stalled")

    monkeypatch.setattr(task_requests.hf_api, "model_info", fake_model_info)

    assert task_requests.get_model_num_params("unsloth/Qwen2-7B") == 7_000_000_000
    assert seen["timeout"] == cst.HF_MODEL_INFO_TIMEOUT_SECONDS
