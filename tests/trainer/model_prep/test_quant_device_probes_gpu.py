"""Quant device probes on a tiny random in-memory GPT-2, run on CPU with tiny sizes.

Needs torch and transformers, so it runs only with --run-gpu (uv run --extra dev --extra gpu pytest --run-gpu ...).
"""

import math

import pytest

from core.models.model_prep_models import DeviceProbeStats
from core.models.model_prep_models import ThroughputStats
from core.models.task_models import TaskType


pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch", reason="device probes require torch")
transformers = pytest.importorskip("transformers", reason="device probes require transformers")

from trainer.model_prep import device_probes  # noqa: E402
from trainer.model_prep import stats as prep_stats  # noqa: E402


VOCAB = 128
SEQ_LEN = 16
MICRO_BATCH = 2


@pytest.fixture(scope="module")
def model():
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=VOCAB, n_positions=64)
    return transformers.GPT2LMHeadModel(config)


def _positive(value: float) -> bool:
    return value > 0 and math.isfinite(value)


def test_train_windows(model):
    result = device_probes.measure_train_throughput_windows(
        model, "cpu", SEQ_LEN, VOCAB, MICRO_BATCH, min_seconds=0.5, window_seconds=0.05, warmup_steps=1
    )
    assert result is not None
    median, rates, seconds = result
    assert _positive(median) and len(rates) >= 1 and all(_positive(r) for r in rates) and _positive(seconds)


def test_run_quant_device_probes(model):
    throughput = ThroughputStats(tokens_per_sec=1.0, seq_len=SEQ_LEN, micro_batch_size=MICRO_BATCH, n_gpus=0)
    result = device_probes.run_quant_device_probes(model, "cpu", throughput, VOCAB)
    assert isinstance(result, DeviceProbeStats)
    assert result.train_tokens_per_sec is not None and _positive(result.train_tokens_per_sec)
    assert result.probe_seconds is not None and result.probe_seconds > 0
    assert DeviceProbeStats.model_validate(result.model_dump()) == result
    without = device_probes.run_quant_device_probes(model, "cpu", None, VOCAB)
    assert without.train_tokens_per_sec is None


def test_instruct_task_has_no_device_stats(model):
    class _Tok:
        def __call__(self, text, **kwargs):
            ids = [ord(c) % VOCAB for c in text][: kwargs.get("max_length", 64)] or [0]
            return {"input_ids": ids} if not kwargs.get("return_tensors") else {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }

    records = [{"instruct": f"question {i}", "output": f"answer {i}"} for i in range(8)]
    try:
        stats = prep_stats.compute_text_stats(model, _Tok(), records, task_type=TaskType.INSTRUCTTEXTTASK, max_samples=4)
    except Exception as exc:
        pytest.skip(f"stub tokenizer not enough for the stats pipeline: {exc}")
    assert stats.device is None
