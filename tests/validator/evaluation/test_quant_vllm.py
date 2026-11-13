import importlib
import json
import sys
from pathlib import Path

import pytest

import validator.evaluation.constants as vcst
from core.models.quant_models import QuantFormat
from tests.validator.evaluation import fake_vllm
from validator.evaluation.errors import EVALUATION_ERROR_RESULT_KEY
from validator.evaluation.evaluators import quant
from validator.evaluation.quant_loss import QuantKernelMismatch


W3A16 = QuantFormat.W3A16_ASYM_G128.spec()
W4A16 = QuantFormat.W4A16_SYM_G128.spec()
W2A16 = QuantFormat.W2A16_ASYM_G64.spec()
W4A8_FP8 = QuantFormat.W4A8_FP8_G128.spec()
HUMMING = vcst.QUANT_EXPECTED_KERNEL_CLASSES[QuantFormat.W3A16_ASYM_G128]
MACHETE = vcst.QUANT_EXPECTED_KERNEL_CLASSES[QuantFormat.W4A16_SYM_G128]
SCORER_MODULES = ("validator.evaluation.quant_vllm", "validator.evaluation.evaluators.single_quant")


def _honest_w3():
    return fake_vllm.quantised(
        name="CompressedTensorsWNA16", kernel="HummingLinearKernel", bits=3, group_size=128, symmetric=False
    )


def _model(**overrides) -> dict:
    """A two-layer decoder as vLLM 0.31 loads an honest W3A16 checkpoint, plus the lm_head."""
    modules = {
        f"model.layers.{i}.{proj}": _honest_w3()
        for i in range(2)
        for proj in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
    }
    modules["lm_head"] = fake_vllm.ParallelLMHead()
    modules.update(overrides)
    return modules


@pytest.fixture
def heuristics():
    """Stand-in for humming.tune._get_heuristics_config: stream-K on, in both return shapes."""

    def get_heuristics_config(layer_config, shape_m=None):
        if isinstance(shape_m, int):
            return {"block_shape": [64, 128, 64], "use_stream_k": True}
        return [(16, 32, {"use_stream_k": True}), (32, 64, {"use_stream_k": True})]

    return get_heuristics_config


@pytest.fixture
def quant_vllm(monkeypatch, heuristics):
    """quant_vllm imported fresh against the fakes; dropped again so no other test sees it."""
    for name, module in fake_vllm.fake_modules(heuristics).items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in SCORER_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in vcst.QUANT_VLLM_ENV:
        monkeypatch.setenv(name, "1")
    yield importlib.import_module("validator.evaluation.quant_vllm")
    for name in SCORER_MODULES:
        sys.modules.pop(name, None)


def test_vllm_env_is_set_in_process_on_import(quant_vllm):
    assert all(quant_vllm.os.environ[name] == value for name, value in vcst.QUANT_VLLM_ENV.items())


def test_honest_layers_pass_and_report_kernels(quant_vllm):
    kernels = quant_vllm.checked_kernel_classes(fake_vllm.FakeLLM(_model()), W3A16, HUMMING)

    assert len(kernels) == 8
    assert set(kernels.values()) == {"HummingLinearKernel"}
    assert "lm_head" not in kernels


def _mismatch(quant_vllm, modules, spec=W3A16, expected=HUMMING) -> str:
    with pytest.raises(QuantKernelMismatch) as exc_info:
        quant_vllm.checked_kernel_classes(fake_vllm.FakeLLM(modules), spec, expected)
    return str(exc_info.value)


def test_bf16_mlps_are_offenders(quant_vllm):
    bf16_mlps = {
        f"model.layers.{i}.{proj}": fake_vllm.unquantised() for i in range(2) for proj in ("mlp.gate_up_proj", "mlp.down_proj")
    }

    reason = _mismatch(quant_vllm, _model(**bf16_mlps))

    assert reason.startswith("4 of 8 linear layers are not W3A16-asym-g128")
    assert "('model.layers.0.mlp.gate_up_proj', 'unquantised (UnquantizedLinearMethod)')" in reason


def test_fully_unquantised_model_fails(quant_vllm):
    reason = _mismatch(quant_vllm, {"model.layers.0.mlp.down_proj": fake_vllm.unquantised()})

    assert "1 of 1 linear layers" in reason


@pytest.mark.parametrize(
    ("scheme_args", "loaded"),
    [
        ({"bits": 6}, "W6 g128"),
        ({"bits": 2, "group_size": 64}, "W2 g64"),
        ({"symmetric": True}, "symmetric=True"),
    ],
    ids=["w6-as-w3", "w2-g64-as-w3", "symmetric-as-asym"],
)
def test_humming_layers_in_another_format_are_offenders(quant_vllm, scheme_args, loaded):
    args = {"name": "CompressedTensorsWNA16", "kernel": "HummingLinearKernel", "bits": 3, "group_size": 128, "symmetric": False}
    impostor = fake_vllm.quantised(**{**args, **scheme_args})

    reason = _mismatch(quant_vllm, _model(**{"model.layers.1.mlp.down_proj": impostor}))

    assert "1 of 8" in reason
    assert loaded in reason


def test_w4a8_int8_through_machete_is_not_w4a16(quant_vllm):
    int8 = {
        "model.layers.0.mlp.down_proj": fake_vllm.quantised(
            name="CompressedTensorsWNA8Int", kernel="MacheteLinearKernel", bits=4, group_size=128, symmetric=True
        )
    }

    assert "activation=int8" in _mismatch(quant_vllm, int8, spec=W4A16, expected=MACHETE)


def test_w4a8_fp8_layers_match_their_spec(quant_vllm):
    fp8 = {
        "model.layers.0.mlp.down_proj": fake_vllm.quantised(
            name="CompressedTensorsW4A8Fp8", kernel="CutlassW4A8LinearKernel", bits=4, group_size=128, symmetric=True
        )
    }
    expected = vcst.QUANT_EXPECTED_KERNEL_CLASSES[QuantFormat.W4A8_FP8_G128]

    assert quant_vllm.checked_kernel_classes(fake_vllm.FakeLLM(fp8), W4A8_FP8, expected) == {
        "model.layers.0.mlp.down_proj": "CutlassW4A8LinearKernel"
    }


def test_right_format_on_an_unexpected_kernel_fails(quant_vllm):
    marlin = {
        "model.layers.0.mlp.down_proj": fake_vllm.quantised(
            name="CompressedTensorsWNA16", kernel="MarlinLinearKernel", bits=4, group_size=128, symmetric=True
        )
    }

    assert "unexpected kernels" in _mismatch(quant_vllm, marlin, spec=W4A16, expected=MACHETE)


def test_unknown_scheme_is_an_offender_not_a_crash(quant_vllm):
    bare = fake_vllm.LinearBase("CompressedTensorsLinearMethod", type("CompressedTensorsW8A8Fp8", (), {})())

    reason = _mismatch(quant_vllm, {"model.layers.0.mlp.down_proj": bare})

    assert "CompressedTensorsW8A8Fp8 WNone gNone symmetric=None activation=unknown" in reason


def test_freeform_reports_without_checking(quant_vllm):
    decoded = {"model.layers.0.mlp.down_proj": fake_vllm.unquantised()}

    assert quant_vllm.checked_kernel_classes(fake_vllm.FakeLLM(decoded), None, None) == {}


def test_kernels_are_read_through_public_apply_model(quant_vllm):
    calls = []

    class RecordingLLM(fake_vllm.FakeLLM):
        def apply_model(self, func):
            calls.append(func)
            return super().apply_model(func)

    quant_vllm.collect_linear_layers(RecordingLLM(_model()))

    assert len(calls) == 1
    assert not hasattr(quant_vllm, "_LOADED_MODEL_PATHS")


def test_launch_formats_map_to_expected_kernels(quant_vllm):
    assert {f: quant_vllm.expected_kernels_for(f.spec()) for f in QuantFormat} == {
        QuantFormat.W4A16_SYM_G128: frozenset({"MacheteLinearKernel"}),
        QuantFormat.W4A8_FP8_G128: frozenset({"CutlassW4A8LinearKernel"}),
        QuantFormat.W3A16_ASYM_G128: frozenset({"HummingLinearKernel"}),
        QuantFormat.W2A16_ASYM_G64: frozenset({"HummingLinearKernel"}),
    }


def test_every_quant_format_has_an_expected_kernel(quant_vllm):
    assert set(vcst.QUANT_EXPECTED_KERNEL_CLASSES) == set(QuantFormat)
    assert quant_vllm.expected_kernels_for(W2A16) == frozenset({"HummingLinearKernel"})


def test_honest_w2a16_layers_pass(quant_vllm):
    w2 = fake_vllm.quantised(
        name="CompressedTensorsWNA16", kernel="HummingLinearKernel", bits=2, group_size=64, symmetric=False
    )
    modules = _model(**{name: w2 for name in _model() if name != "lm_head"})

    assert quant_vllm.checked_kernel_classes(fake_vllm.FakeLLM(modules), W2A16, HUMMING)


def test_stream_k_patch_forces_stream_k_off(quant_vllm):
    quant_vllm.disable_humming_stream_k()
    patched = quant_vllm.humming_tune._get_heuristics_config

    assert patched("layer", shape_m=16) == {"block_shape": [64, 128, 64], "use_stream_k": False}
    assert [entry[2]["use_stream_k"] for entry in patched("layer")] == [False, False]


def test_missing_stream_k_patch_target_raises(quant_vllm, monkeypatch):
    monkeypatch.delattr(quant_vllm.humming_tune, "_get_heuristics_config")

    with pytest.raises(RuntimeError, match="humming-kernels 0.1.16"):
        quant_vllm.disable_humming_stream_k()


@pytest.fixture
def single_quant(quant_vllm, monkeypatch, tmp_path):
    single_quant = importlib.import_module("validator.evaluation.evaluators.single_quant")
    monkeypatch.setattr(quant, "CONTAINER_EVAL_RESULTS_PATH", str(tmp_path / "evaluation_results.json"))
    monkeypatch.setattr(single_quant, "record_base_model_params", lambda _model: None)
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(json.dumps({"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}) + "\n")
    args = quant.QuantEvaluationArgs(
        dataset=str(dataset), original_model="org/base", repo="org/repo", quant_format=QuantFormat.W3A16_ASYM_G128
    )
    return single_quant, args


def test_run_patches_stream_k_before_scoring(single_quant, monkeypatch):
    module, args = single_quant
    order = []
    monkeypatch.setattr(module, "disable_humming_stream_k", lambda: order.append("patch"))

    def fake_evaluate_repo(**kwargs):
        order.append("evaluate")
        assert kwargs["quant_spec"] == W3A16
        return {"is_finetune": True, "eval_loss": 1.0}

    monkeypatch.setattr(module, "evaluate_repo", fake_evaluate_repo)

    module.run(args)

    assert order == ["patch", "evaluate"]
    assert quant.load_results() == {"org/repo": {"is_finetune": True, "eval_loss": 1.0}}


def test_missing_patch_target_is_an_evaluation_error_not_a_scored_fail(single_quant, monkeypatch):
    module, args = single_quant
    monkeypatch.delattr(sys.modules["humming.tune"], "_get_heuristics_config")
    monkeypatch.setattr(module, "evaluate_repo", lambda **_kwargs: pytest.fail("scored without the patch"))

    module.run(args)

    result = quant.load_results()["org/repo"]
    assert set(result) == {EVALUATION_ERROR_RESULT_KEY}
    assert "_get_heuristics_config is missing" in result[EVALUATION_ERROR_RESULT_KEY]


def test_the_reference_run_saves_under_its_key_with_no_spec(single_quant, monkeypatch):
    module, args = single_quant
    reference_args = args.model_copy(update={"quant_format": None, "result_key": vcst.QUANT_REFERENCE_RESULT_KEY})
    monkeypatch.setattr(module, "disable_humming_stream_k", lambda: None)

    def fake_evaluate_repo(**kwargs):
        assert kwargs["quant_spec"] is None
        return {"is_finetune": True, "eval_loss": 1.0}

    monkeypatch.setattr(module, "evaluate_repo", fake_evaluate_repo)

    module.run(reference_args)

    assert quant.load_results() == {vcst.QUANT_REFERENCE_RESULT_KEY: {"is_finetune": True, "eval_loss": 1.0}}


def test_the_engine_reads_only_weights_and_config_from_the_repo(quant_vllm):
    engine = quant_vllm.VllmPromptLogprobEngine("org/miner-repo", max_prompt_len=99, tokenizer="org/base")

    kwargs = engine.llm.kwargs
    assert kwargs["model"] == "org/miner-repo"
    assert kwargs["tokenizer"] == "org/base"
    assert (kwargs["tokenizer_mode"], kwargs["config_format"], kwargs["load_format"], kwargs["generation_config"]) == (
        "hf",
        "hf",
        "safetensors",
        "vllm",
    )
    assert kwargs["max_model_len"] == kwargs["max_num_batched_tokens"] == 100
    assert "chat_template" not in kwargs


def test_resolved_model_config_is_vllms_hf_view_of_that_file_alone(quant_vllm, monkeypatch):
    raw = {"model_type": "qwen2", "rms_norm_eps": 1e-06, "rope_theta": 1000000.0}
    calls = []

    def get_config(model, **kwargs):
        model_dir = Path(model)
        calls.append(([path.name for path in model_dir.iterdir()], json.loads((model_dir / "config.json").read_text()), kwargs))
        return fake_vllm.FakeHfConfig({"model_type": "qwen2", "rope_parameters": {"rope_theta": 1000000.0}})

    monkeypatch.setattr(quant_vllm, "get_config", get_config)

    assert quant_vllm.resolved_model_config(raw) == {"model_type": "qwen2", "rope_parameters": {"rope_theta": 1000000.0}}
    assert calls == [(["config.json"], raw, {"trust_remote_code": False, "config_format": "hf"})]


def test_run_checks_the_config_against_the_input_model_and_pins_its_tokenizer(single_quant, monkeypatch):
    """A miner's tokenizer files (a changed chat template, special tokens) never reach the engine: it is built
    with the input model's tokenizer, and the config is resolved against the input model."""
    module, args = single_quant
    monkeypatch.setattr(module, "disable_humming_stream_k", lambda: None)
    built = []
    monkeypatch.setattr(module, "VllmPromptLogprobEngine", lambda path, max_len, tokenizer: built.append((path, tokenizer)))

    def fake_evaluate_repo(**kwargs):
        check = kwargs["config_check"]
        assert check.base_model == "org/base"
        assert check.resolve is module.resolved_model_config
        kwargs["engine_factory"]("org/repo")
        return {"is_finetune": True, "eval_loss": 1.0}

    monkeypatch.setattr(module, "evaluate_repo", fake_evaluate_repo)

    module.run(args)

    assert built == [("org/repo", "org/base")]
