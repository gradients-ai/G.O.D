import json
import logging
import sys
import urllib.error
from types import ModuleType
from types import SimpleNamespace
from uuid import uuid4

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

import core.constants.docker as docker_cst
import validator.evaluation.constants as vcst
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import EvaluationResultText
from core.models.quant_models import QuantEvalGpu
from core.models.quant_models import QuantFormat
from validator.evaluation import docker_evaluation
from validator.evaluation.evaluators import quant
from validator.evaluation.quant_artifacts import DecodedModelCorrupt
from validator.evaluation.quant_artifacts import decoded_model_prefix
from validator.evaluation.quant_loss import QuantEvalExample
from validator.evaluation.quant_loss import QuantKernelMismatch


REPO = "org/repo-a"
OTHER_REPO = "org/repo-b"
EXAMPLES = [
    QuantEvalExample(input_ids=[1, 2, 3, 4], labels=[-100, -100, 3, 4]),
    QuantEvalExample(input_ids=[5, 6, 7], labels=[-100, 6, 7]),
]
QUANTISED_LAYERS = {"model.layers.0.self_attn.q_proj": "MacheteLinearKernel"}
H100_SXM = QuantEvalGpu(name="NVIDIA H100 80GB HBM3", sm_count=132)
H100_PCIE = QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114)
BASE = "org/base"
# The input model's config.json, and configs resolved as quant_vllm.resolved_model_config returns them.
BASE_RAW = {"model_type": "qwen2", "rms_norm_eps": 1e-06, "rope_theta": 1000000.0}
BASE_RESOLVED = {
    "model_type": "qwen2",
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
}
SAME_CONFIG = quant.FixedConfigCheck(base_model=BASE, resolve=lambda _config: BASE_RESOLVED)
W4A16 = QuantFormat.W4A16_SYM_G128.spec()
W4A16_CONFIG = {
    "quantization_config": {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "group",
                    "group_size": 128,
                    "symmetric": True,
                    "actorder": None,
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
    }
}


async def _run_quant_dispatch(monkeypatch, quant_spec, quant_format, decoded_model_urls=None, reference_only=False) -> dict:
    captured = {}

    async def fake_load_eval_pair_state_for_models(*_args, **_kwargs):
        return {}, {REPO: "hk_a"}

    async def fake_run_remote_eval_repos(**kwargs):
        captured.update(kwargs)
        return {REPO: {REPO: {"is_finetune": True, "eval_loss": 1.25}}}

    monkeypatch.setattr(docker_evaluation, "load_eval_pair_state_for_models", fake_load_eval_pair_state_for_models)
    monkeypatch.setattr(docker_evaluation, "run_remote_eval_repos", fake_run_remote_eval_repos)

    results = await docker_evaluation.run_evaluation_basilica_quant(
        dataset="https://bucket.example/test_data.jsonl",
        models=[REPO],
        original_model="org/base",
        dataset_type=InstructTextDatasetType(),
        file_format=FileFormat.S3,
        num_gpus=1,
        quant_spec=quant_spec,
        quant_format=quant_format,
        task_id=uuid4(),
        psql_db=object(),
        decoded_model_urls=decoded_model_urls,
        reference_only=reference_only,
    )
    assert results.results[REPO].eval_loss == 1.25
    return captured


@pytest.mark.asyncio
async def test_fixed_track_dispatch_targets_h100_quant_image_with_spec(monkeypatch):
    spec = QuantFormat.W4A16_SYM_G128.spec()

    captured = await _run_quant_dispatch(monkeypatch, spec, QuantFormat.W4A16_SYM_G128)

    env = captured["build_env_for_repo"](REPO)
    assert captured["gpu_models"] == vcst.QUANT_EVAL_GPU_MODELS
    assert captured["runpod_instance_types"] == ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"]
    assert docker_cst.QUANT_EVAL_ALLOW_ANY_GPU_ENV not in env
    assert docker_cst.QUANT_REFERENCE_ONLY_ENV not in env
    assert captured["image"] == docker_cst.VALIDATOR_DOCKER_IMAGE_QUANT
    assert "validator.evaluation.evaluators.quant" in captured["source"]
    assert env[docker_cst.QUANT_SPEC_ENV] == spec.model_dump_json()
    assert env[docker_cst.QUANT_FORMAT_ENV] == QuantFormat.W4A16_SYM_G128.value
    assert env["MODELS"] == REPO
    assert env["DATASET_URL"] == "https://bucket.example/test_data.jsonl"


@pytest.mark.asyncio
async def test_reference_only_dispatch_also_offers_a100s_and_flags_the_container(monkeypatch):
    captured = await _run_quant_dispatch(monkeypatch, None, None, reference_only=True)

    env = captured["build_env_for_repo"](REPO)
    assert env[docker_cst.QUANT_REFERENCE_ONLY_ENV] == "1"
    assert captured["gpu_models"] == ["H100", "A100"]
    assert captured["runpod_instance_types"] == [
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H100 PCIe",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100 80GB PCIe",
    ]
    assert docker_cst.QUANT_EVAL_ALLOW_ANY_GPU_ENV not in env


@pytest.mark.asyncio
async def test_freeform_dispatch_omits_quant_spec(monkeypatch):
    captured = await _run_quant_dispatch(monkeypatch, None, None)

    env = captured["build_env_for_repo"](REPO)
    assert docker_cst.QUANT_SPEC_ENV not in env
    assert docker_cst.QUANT_FORMAT_ENV not in env
    assert captured["gpu_models"] == vcst.QUANT_EVAL_GPU_MODELS
    assert captured["runpod_instance_types"] == vcst.QUANT_EVAL_RUNPOD_INSTANCE_TYPES
    assert docker_cst.QUANT_EVAL_ALLOW_ANY_GPU_ENV not in env


@pytest.mark.asyncio
async def test_decoded_model_urls_reach_only_their_own_deployment(monkeypatch):
    urls = {REPO: {"index.json": "https://s3/i", "model.safetensors": "https://s3/m"}}
    captured = await _run_quant_dispatch(monkeypatch, None, None, decoded_model_urls=urls)

    env = captured["build_env_for_repo"](REPO)
    assert json.loads(env[docker_cst.DECODED_MODEL_URLS_ENV]) == urls
    assert docker_cst.DECODED_MODEL_URLS_ENV not in captured["build_env_for_repo"](OTHER_REPO)


@pytest.mark.asyncio
async def test_fixed_track_env_has_no_decoded_model_urls(monkeypatch):
    captured = await _run_quant_dispatch(monkeypatch, QuantFormat.W4A16_SYM_G128.spec(), QuantFormat.W4A16_SYM_G128)

    assert docker_cst.DECODED_MODEL_URLS_ENV not in captured["build_env_for_repo"](REPO)


class FakeMinio:
    """Lists objects per prefix and presigns each name; records every list_objects call."""

    def __init__(self, objects_by_prefix: dict[str, list[str]]):
        self.objects_by_prefix = objects_by_prefix
        self.list_calls = []

    async def list_objects(self, bucket, prefix, recursive):
        self.list_calls.append({"bucket": bucket, "prefix": prefix, "recursive": recursive})
        return iter(SimpleNamespace(object_name=name) for name in self.objects_by_prefix.get(prefix, []))

    async def get_presigned_url(self, bucket, name):
        return f"https://{bucket}/{name}?signed"


class RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


@pytest.mark.asyncio
async def test_decoded_model_urls_only_for_prefixes_with_an_index(monkeypatch):
    task_id = uuid4()
    done = decoded_model_prefix(task_id, "hk_done")
    partial = decoded_model_prefix(task_id, "hk_partial")
    minio = FakeMinio(
        {
            f"{done}/": [f"{done}/{docker_cst.DECODED_MODEL_INDEX_NAME}", f"{done}/model-00001.safetensors"],
            f"{partial}/": [f"{partial}/model-00001.safetensors"],
        }
    )
    logger = RecordingLogger()
    monkeypatch.setattr(docker_evaluation, "async_minio_client", minio)
    monkeypatch.setattr(docker_evaluation, "logger", logger)

    urls = await docker_evaluation.decoded_model_urls_for(task_id, {"hk_done": REPO, "hk_partial": OTHER_REPO}, "bucket")

    assert urls == {
        REPO: {
            docker_cst.DECODED_MODEL_INDEX_NAME: f"https://bucket/{done}/{docker_cst.DECODED_MODEL_INDEX_NAME}?signed",
            "model-00001.safetensors": f"https://bucket/{done}/model-00001.safetensors?signed",
        }
    }
    assert len(logger.warnings) == 1 and "hk_partial" in logger.warnings[0]
    assert all(call["recursive"] is False for call in minio.list_calls)


class FakeEngine:
    def __init__(self, logprob: float | None = -0.5):
        self.logprob = logprob

    def prompt_logprobs(self, token_ids: list[int]) -> list[float | None]:
        return [None] + [self.logprob] * (len(token_ids) - 1)


def _layer_check(kernel_classes=QUANTISED_LAYERS, mismatch: str | None = None):
    def check(_engine) -> dict[str, str]:
        if mismatch is not None:
            raise QuantKernelMismatch(mismatch)
        return kernel_classes

    return check


def _evaluate(engine_factory, layer_check=None, config=W4A16_CONFIG) -> dict:
    return quant.evaluate_repo(
        repo=REPO,
        examples=EXAMPLES,
        engine_factory=engine_factory,
        layer_check=layer_check or _layer_check(),
        quant_spec=W4A16,
        config_check=SAME_CONFIG,
        config_loader=lambda _repo: config,
    )


def test_evaluate_repo_returns_success_dict():
    result = _evaluate(lambda _repo: FakeEngine())

    assert result["is_finetune"] is True
    assert result["eval_loss"] == pytest.approx(0.5)
    assert result["per_example_losses"] == pytest.approx([0.5, 0.5])
    assert len(result["eval_set_fingerprint"]) == 32
    assert result["kernel_classes"] == QUANTISED_LAYERS
    assert "quant_failure" not in result


def test_freeform_skips_config_check():
    def no_config(_repo):
        raise AssertionError("freeform must not read config.json")

    result = quant.evaluate_repo(
        repo=REPO,
        examples=EXAMPLES,
        engine_factory=lambda _repo: FakeEngine(),
        layer_check=_layer_check(kernel_classes={}),
        config_loader=no_config,
    )

    assert result["eval_loss"] == pytest.approx(0.5)


def test_config_mismatch_is_a_scored_fail_before_loading():
    loaded = []
    w3_config = json.loads(json.dumps(W4A16_CONFIG))
    w3_config["quantization_config"]["config_groups"]["group_0"]["weights"]["num_bits"] = 3

    result = _evaluate(lambda repo: loaded.append(repo), config=w3_config)

    assert loaded == []
    assert result["eval_loss"] is None
    assert result["quant_failure"].startswith("quantization_config is not W4A16-sym-g128")
    assert "num_bits is 3" in result["quant_failure"]


def _resolver(submitted: dict | Exception, base: dict | Exception = BASE_RESOLVED):
    """FixedConfigCheck.resolve: base for the input model's config.json, submitted for the repo's (raised if an
    exception)."""

    def resolve(config: dict) -> dict:
        resolved = base if config is BASE_RAW else submitted
        if isinstance(resolved, Exception):
            raise resolved
        return resolved

    return resolve


def _evaluate_with_resolved(submitted: dict | Exception, loaded: list, base: dict | Exception = BASE_RESOLVED) -> dict:
    return quant.evaluate_repo(
        repo=REPO,
        examples=EXAMPLES,
        engine_factory=lambda repo: loaded.append(repo) or FakeEngine(),
        layer_check=_layer_check(),
        quant_spec=W4A16,
        config_check=quant.FixedConfigCheck(base_model=BASE, resolve=_resolver(submitted, base)),
        config_loader=lambda repo: BASE_RAW if repo == BASE else W4A16_CONFIG,
    )


@pytest.mark.parametrize(
    ("submitted", "reason"),
    [
        ({**BASE_RESOLVED, "rms_norm_eps": 1e-05}, "config.rms_norm_eps is 1e-05, the input model has 1e-06"),
        (
            {**BASE_RESOLVED, "rope_parameters": {"rope_theta": 2000000.0, "rope_type": "default"}},
            "config.rope_parameters.rope_theta is 2000000.0, the input model has 1000000.0",
        ),
    ],
    ids=["rms_norm_eps", "rope_theta"],
)
def test_config_drift_from_the_input_model_is_a_scored_fail_before_loading(submitted, reason):
    loaded = []

    result = _evaluate_with_resolved(submitted, loaded)

    assert loaded == []
    assert result["eval_loss"] is None
    assert result["quant_failure"] == f"config differs from the input model: {reason}"


def test_a_config_resolving_to_the_input_models_is_scored():
    loaded = []
    resolved = {**BASE_RESOLVED, "quantization_config": W4A16_CONFIG["quantization_config"], "transformers_version": "5.17.0"}

    result = _evaluate_with_resolved(resolved, loaded)

    assert loaded == [REPO]
    assert result["eval_loss"] == pytest.approx(0.5)


@pytest.mark.parametrize("error", [ValueError, RuntimeError], ids=["value-error", "other"])
def test_a_submitted_config_that_does_not_resolve_is_a_scored_fail(error):
    loaded = []

    result = _evaluate_with_resolved(error("rope_parameters is invalid"), loaded)

    assert loaded == []
    assert result["quant_failure"] == "config.json does not load: rope_parameters is invalid"


def test_an_input_model_config_that_does_not_resolve_is_an_evaluation_error():
    with pytest.raises(ValueError, match="base config"):
        _evaluate_with_resolved(BASE_RESOLVED, [], base=ValueError("base config"))


def test_an_unreachable_input_model_config_is_an_evaluation_error():
    def config_loader(repo):
        if repo == BASE:
            raise LocalEntryNotFoundError("hub unreachable")
        return W4A16_CONFIG

    with pytest.raises(LocalEntryNotFoundError):
        quant.evaluate_repo(
            repo=REPO,
            examples=EXAMPLES,
            engine_factory=lambda _repo: FakeEngine(),
            layer_check=_layer_check(),
            quant_spec=W4A16,
            config_check=SAME_CONFIG,
            config_loader=config_loader,
        )


def test_fixed_track_without_the_input_model_is_an_evaluation_error():
    with pytest.raises(ValueError, match="config_check"):
        quant.evaluate_repo(
            repo=REPO,
            examples=EXAMPLES,
            engine_factory=lambda _repo: FakeEngine(),
            layer_check=_layer_check(),
            quant_spec=W4A16,
            config_loader=lambda _repo: W4A16_CONFIG,
        )


def test_unreadable_config_is_a_scored_fail():
    def missing(_repo):
        raise OSError("404 config.json")

    result = quant.evaluate_repo(
        repo=REPO,
        examples=EXAMPLES,
        engine_factory=lambda _repo: FakeEngine(),
        layer_check=_layer_check(),
        quant_spec=W4A16,
        config_check=SAME_CONFIG,
        config_loader=missing,
    )

    assert result["quant_failure"] == "config.json unreadable: 404 config.json"


def _hub_config_file(monkeypatch, tmp_path, text: str):
    """quant.hf_hub_download returning a config.json holding text, recording the repo and token asked for."""
    downloads = []
    path = tmp_path / "config.json"
    path.write_text(text)

    def hf_hub_download(repo, filename, token=None):
        downloads.append((repo, filename, token))
        return str(path)

    monkeypatch.setattr(quant, "hf_hub_download", hf_hub_download)
    return downloads


def test_load_hub_config_parses_the_downloaded_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_x")
    downloads = _hub_config_file(monkeypatch, tmp_path, json.dumps(W4A16_CONFIG))

    assert quant.load_hub_config(REPO) == W4A16_CONFIG
    assert downloads == [(REPO, "config.json", "hf_x")]


def test_load_hub_config_refuses_a_huge_file_before_parsing_it(monkeypatch, tmp_path):
    """A multi-GB config.json would exhaust the shared eval container's memory in json.loads; the cap turns it into a
    scored fail for the submission (ValueError is what _config_failure reports as unreadable)."""
    _hub_config_file(monkeypatch, tmp_path, "{" + " " * vcst.QUANT_CONFIG_MAX_BYTES + "}")

    with pytest.raises(ValueError, match="over the"):
        quant.load_hub_config(REPO)


def test_a_huge_submitted_config_is_a_scored_fail(monkeypatch, tmp_path):
    _hub_config_file(monkeypatch, tmp_path, "{" + " " * vcst.QUANT_CONFIG_MAX_BYTES + "}")

    result = quant.evaluate_repo(
        repo=REPO,
        examples=EXAMPLES,
        engine_factory=lambda _repo: pytest.fail("loaded a repo whose config.json is over the cap"),
        layer_check=_layer_check(),
        quant_spec=W4A16,
        config_check=SAME_CONFIG,
    )

    assert result["eval_loss"] is None
    assert result["quant_failure"].startswith("config.json unreadable: config.json is 1048578 bytes, over the 1048576 byte cap")


@pytest.mark.parametrize("config", [[], "x"], ids=["list", "str"])
def test_non_object_config_is_a_scored_fail(config):
    result = _evaluate(lambda _repo: FakeEngine(), config=config)

    assert result["eval_loss"] is None
    assert result["quant_failure"] == "quantization_config is not W4A16-sym-g128: config.json is not a JSON object"


def test_unreachable_hub_is_an_evaluation_error():
    def offline(_repo):
        raise LocalEntryNotFoundError("hub unreachable and config.json not cached")

    with pytest.raises(LocalEntryNotFoundError):
        quant.evaluate_repo(
            repo=REPO,
            examples=EXAMPLES,
            engine_factory=lambda _repo: FakeEngine(),
            layer_check=_layer_check(),
            quant_spec=W4A16,
            config_check=SAME_CONFIG,
            config_loader=offline,
        )


def _raise_on_load(_repo):
    raise OSError("missing config.json")


@pytest.mark.parametrize(
    ("engine_factory", "layer_check", "reason"),
    [
        (_raise_on_load, _layer_check(), "failed to load"),
        (lambda _repo: FakeEngine(), _layer_check(mismatch="model.layers.0.mlp is unquantised"), "model.layers.0.mlp"),
        (lambda _repo: FakeEngine(float("nan")), _layer_check(), "not finite"),
        (lambda _repo: FakeEngine(None), _layer_check(), "no logprob"),
    ],
    ids=["load-error", "layer-mismatch", "non-finite-loss", "missing-logprob"],
)
def test_evaluate_repo_returns_scored_fail(engine_factory, layer_check, reason):
    result = _evaluate(engine_factory, layer_check=layer_check)

    assert set(result) == {"is_finetune", "eval_loss", "quant_failure", "eval_gpu"}
    assert result["is_finetune"] is True
    assert result["eval_loss"] is None
    assert reason in result["quant_failure"]
    assert len(result["quant_failure"]) <= vcst.QUANT_FAILURE_REASON_MAX_CHARS


@pytest.mark.parametrize(
    "error",
    [TimeoutError("hub read timed out"), ConnectionError("connection reset"), urllib.error.URLError("name resolution failed")],
    ids=["timeout", "connection", "url-error"],
)
def test_an_infrastructure_error_loading_the_engine_is_an_evaluation_error_not_a_scored_fail(error):
    def raise_infra(_repo):
        raise error

    result = _evaluate(raise_infra)

    assert list(result) == ["evaluation_error"]
    assert "quant_failure" not in result


def test_engine_nan_on_one_example_is_a_scored_fail():
    class NanOnLastEngine:
        def prompt_logprobs(self, token_ids: list[int]) -> list[float | None]:
            logprob = float("nan") if token_ids == EXAMPLES[-1].input_ids else -0.5
            return [None] + [logprob] * (len(token_ids) - 1)

    result = _evaluate(lambda _repo: NanOnLastEngine())

    assert result["eval_loss"] is None
    assert "not finite" in result["quant_failure"]


def test_validator_side_value_error_is_not_a_scored_fail():
    class MisconfiguredEngine:
        def prompt_logprobs(self, token_ids: list[int]) -> list[float | None]:
            raise ValueError("prompt is longer than max_model_len")

    with pytest.raises(ValueError, match="max_model_len"):
        _evaluate(lambda _repo: MisconfiguredEngine())


def test_scored_fail_reason_is_truncated():
    def raise_long(_repo):
        raise RuntimeError("x" * 1000)

    result = _evaluate(raise_long)

    assert len(result["quant_failure"]) == vcst.QUANT_FAILURE_REASON_MAX_CHARS


def test_main_exits_when_models_missing(monkeypatch):
    monkeypatch.setenv("DATASET", "/tmp/test.jsonl")
    monkeypatch.setenv("ORIGINAL_MODEL", "org/base")
    monkeypatch.setenv("MODELS", " , ")

    with pytest.raises(SystemExit) as exc_info:
        quant.main()

    assert exc_info.value.code == 1


def test_evaluate_repo_loads_model_path_and_keys_by_repo():
    loaded = []

    def engine_factory(path):
        loaded.append(path)
        return FakeEngine()

    result = quant.evaluate_repo(
        repo=REPO,
        examples=EXAMPLES,
        engine_factory=engine_factory,
        layer_check=_layer_check(kernel_classes={}),
        model_path="/tmp/decoded-x",
    )

    assert loaded == ["/tmp/decoded-x"]
    assert result["eval_loss"] == pytest.approx(0.5)


@pytest.fixture
def quant_main_env(monkeypatch, tmp_path):
    """Env for quant.main on the freeform track; returns (subprocess calls, results file path)."""
    results_path = tmp_path / "out" / "evaluation_results.json"
    urls = {REPO: {"index.json": "https://s3/i", "model.safetensors": "https://s3/m"}}
    monkeypatch.setenv("DATASET", str(tmp_path / "test.jsonl"))
    monkeypatch.setenv("ORIGINAL_MODEL", "org/base")
    monkeypatch.setenv("MODELS", f"{REPO},{OTHER_REPO}")
    monkeypatch.setenv(docker_cst.DECODED_MODEL_URLS_ENV, json.dumps(urls))
    monkeypatch.setattr(quant, "CONTAINER_EVAL_RESULTS_PATH", str(results_path))
    monkeypatch.setattr(quant, "probe_gpu", lambda: H100_SXM)
    calls = []
    monkeypatch.setattr(quant.subprocess, "run", lambda args, check: calls.append(args))
    return calls, results_path


def test_main_routes_materialised_path_into_subprocess(monkeypatch, quant_main_env):
    calls, _ = quant_main_env
    materialised = []

    def fake_materialise(original_model, urls, dest, hf_token):
        materialised.append((original_model, urls))
        return dest

    monkeypatch.setattr(quant, "materialise_decoded_model", fake_materialise)

    quant.main()

    args_by_repo = {args.repo: args for args in (quant.QuantEvaluationArgs.model_validate_json(c[-1]) for c in calls)}
    assert materialised == [("org/base", {"index.json": "https://s3/i", "model.safetensors": "https://s3/m"})]
    assert args_by_repo[REPO].model_path is not None
    assert "decoded-" in args_by_repo[REPO].model_path
    assert args_by_repo[OTHER_REPO].model_path is None


def test_main_records_corrupt_decoded_model_as_scored_fail(monkeypatch, quant_main_env):
    calls, results_path = quant_main_env

    def corrupt(*_args):
        raise DecodedModelCorrupt("model.safetensors does not match its index entry " + "x" * 500)

    monkeypatch.setattr(quant, "materialise_decoded_model", corrupt)

    quant.main()

    result = json.loads(results_path.read_text())[REPO]
    assert result["eval_loss"] is None
    assert result["quant_failure"].startswith("decoded model: model.safetensors")
    assert len(result["quant_failure"]) == vcst.QUANT_FAILURE_REASON_MAX_CHARS
    assert [quant.QuantEvaluationArgs.model_validate_json(c[-1]).repo for c in calls] == [OTHER_REPO]


@pytest.mark.parametrize(("safetensors", "recorded"), [(SimpleNamespace(total=1_543_714_304), 1_543_714_304), (None, None)])
def test_record_base_model_params_from_hub_metadata(monkeypatch, tmp_path, safetensors, recorded):
    monkeypatch.setattr(quant, "CONTAINER_EVAL_RESULTS_PATH", str(tmp_path / "evaluation_results.json"))
    monkeypatch.setattr(
        quant, "HfApi", lambda: SimpleNamespace(model_info=lambda _model: SimpleNamespace(safetensors=safetensors))
    )
    quant.save_results({REPO: {"eval_loss": 1.0}})

    quant.record_base_model_params("org/base")

    assert quant.load_results().get(quant.MODEL_PARAMS_COUNT_KEY) == recorded
    assert quant.load_results()[REPO] == {"eval_loss": 1.0}


A100 = QuantEvalGpu(name="NVIDIA A100-SXM4-80GB", sm_count=108)


@pytest.mark.parametrize(("gpu", "sku"), [(H100_SXM, "H100 SXM"), (H100_PCIE, "H100 PCIe")])
def test_check_gpu_accepts_both_h100_skus_and_records_the_sku(gpu, sku):
    checked = quant.check_gpu(gpu)

    assert (checked.name, checked.sm_count, checked.sku, checked.override) == (gpu.name, gpu.sm_count, sku, False)


@pytest.mark.parametrize(
    "gpu",
    [
        A100,
        QuantEvalGpu(name="NVIDIA H100 NVL", sm_count=132),
        # The name alone is not enough: a MIG slice or an odd part reports fewer SMs.
        QuantEvalGpu(name="NVIDIA H100 80GB HBM3", sm_count=16),
        QuantEvalGpu(name="NVIDIA L40S", sm_count=114),
    ],
)
def test_check_gpu_fails_closed_on_any_other_gpu(gpu):
    with pytest.raises(quant.QuantGpuMismatch, match=f"this container has {gpu.name} with {gpu.sm_count} SMs"):
        quant.check_gpu(gpu)


def test_reference_skus_admit_80gb_a100s_under_their_own_label():
    checked = quant.check_gpu(A100, allowed=vcst.QUANT_REFERENCE_ALLOWED_SKUS)

    assert (checked.sku, checked.override, checked.reference_key) == ("A100 SXM", False, "A100 SXM")
    assert quant.check_gpu(H100_SXM, allowed=vcst.QUANT_REFERENCE_ALLOWED_SKUS).sku == "H100 SXM"


def test_reference_skus_still_refuse_a_40gb_a100():
    with pytest.raises(quant.QuantGpuMismatch):
        quant.check_gpu(QuantEvalGpu(name="NVIDIA A100-SXM4-40GB", sm_count=108), allowed=vcst.QUANT_REFERENCE_ALLOWED_SKUS)


def test_entries_never_get_the_reference_skus():
    assert quant.allowed_skus_for_run(False, [REPO], REPO, None, {}) == vcst.QUANT_EVAL_ALLOWED_SKUS
    assert quant.allowed_skus_for_run(True, [REPO], REPO, None, {}) == vcst.QUANT_REFERENCE_ALLOWED_SKUS


@pytest.mark.parametrize(
    ("repos", "reference", "spec", "decoded"),
    [
        ([REPO], None, None, {}),
        ([REPO], OTHER_REPO, None, {}),
        ([REPO, OTHER_REPO], REPO, None, {}),
        ([REPO], REPO, QuantFormat.W4A16_SYM_G128.spec(), {}),
        ([REPO], REPO, None, {REPO: {"index.json": "https://s3/i"}}),
    ],
)
def test_reference_only_on_anything_but_a_bare_reference_fails_closed(repos, reference, spec, decoded):
    with pytest.raises(quant.QuantReferenceOnlyMisuse):
        quant.allowed_skus_for_run(True, repos, reference, spec, decoded)


def test_main_measures_a_bare_reference_on_an_a100(monkeypatch, quant_main_env):
    calls, _ = quant_main_env
    monkeypatch.setenv("MODELS", REPO)
    monkeypatch.delenv(docker_cst.DECODED_MODEL_URLS_ENV)
    monkeypatch.setenv(docker_cst.QUANT_REFERENCE_MODEL_ENV, REPO)
    monkeypatch.setenv(docker_cst.QUANT_REFERENCE_ONLY_ENV, "1")
    monkeypatch.setattr(quant, "probe_gpu", lambda: A100)

    quant.main()

    gpus = [quant.QuantEvaluationArgs.model_validate_json(c[-1]).eval_gpu for c in calls]
    assert [(g.sku, g.override) for g in gpus] == [("A100 SXM", False)]


def test_main_refuses_reference_only_on_an_entry_before_loading(monkeypatch, quant_main_env):
    calls, results_path = quant_main_env
    monkeypatch.setenv(docker_cst.QUANT_REFERENCE_MODEL_ENV, "org/base")
    monkeypatch.setenv(docker_cst.QUANT_REFERENCE_ONLY_ENV, "1")
    monkeypatch.setattr(quant, "probe_gpu", lambda: A100)
    monkeypatch.setattr(quant, "materialise_decoded_model", lambda *_args: pytest.fail("loaded a model"))

    with pytest.raises(quant.QuantReferenceOnlyMisuse):
        quant.main()

    assert calls == []
    assert not results_path.exists()


def test_check_gpu_override_admits_an_unlisted_gpu_loudly(caplog):
    with caplog.at_level("WARNING"):
        checked = quant.check_gpu(A100, allow_any=True)

    assert (checked.sku, checked.override) == (None, True)
    assert "QUANT_EVAL_ALLOW_ANY_GPU OVERRIDE" in caplog.text


def test_check_gpu_override_does_not_relabel_an_allowed_sku():
    assert quant.check_gpu(H100_PCIE, allow_any=True).override is False


def test_main_fails_the_container_on_the_wrong_gpu_before_loading(monkeypatch, quant_main_env):
    calls, results_path = quant_main_env
    monkeypatch.setattr(quant, "probe_gpu", lambda: A100)
    monkeypatch.setattr(quant, "materialise_decoded_model", lambda *_args: pytest.fail("loaded a model"))

    with pytest.raises(quant.QuantGpuMismatch):
        quant.main()

    assert calls == []
    assert not results_path.exists()


@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_main_override_needs_the_literal_one(monkeypatch, quant_main_env, value):
    monkeypatch.setenv(docker_cst.QUANT_EVAL_ALLOW_ANY_GPU_ENV, value)
    monkeypatch.setattr(quant, "probe_gpu", lambda: A100)

    with pytest.raises(quant.QuantGpuMismatch):
        quant.main()


def test_main_override_scores_on_an_unlisted_gpu(monkeypatch, quant_main_env):
    calls, _ = quant_main_env
    monkeypatch.setenv(docker_cst.QUANT_EVAL_ALLOW_ANY_GPU_ENV, "1")
    monkeypatch.setattr(quant, "probe_gpu", lambda: A100)
    monkeypatch.setattr(quant, "materialise_decoded_model", lambda _model, _urls, dest, _token: dest)

    quant.main()

    gpus = [quant.QuantEvaluationArgs.model_validate_json(c[-1]).eval_gpu for c in calls]
    assert [(g.name, g.sku, g.override) for g in gpus] == [(A100.name, None, True)] * 2


@pytest.mark.parametrize(("gpu", "sku"), [(H100_SXM, "H100 SXM"), (H100_PCIE, "H100 PCIe")])
def test_main_passes_the_checked_gpu_to_every_repo(monkeypatch, quant_main_env, gpu, sku):
    calls, _ = quant_main_env
    monkeypatch.setattr(quant, "probe_gpu", lambda: gpu)
    monkeypatch.setattr(quant, "materialise_decoded_model", lambda _model, _urls, dest, _token: dest)

    quant.main()

    expected = gpu.model_copy(update={"sku": sku})
    assert [quant.QuantEvaluationArgs.model_validate_json(c[-1]).eval_gpu for c in calls] == [expected, expected]


def test_dispatch_logs_an_override_result_as_an_error(caplog):
    gpu = A100.model_copy(update={"override": True})
    processed = docker_evaluation.DockerEvaluationResults(
        results={REPO: EvaluationResultText(is_finetune=True, eval_loss=1.0, eval_gpu=gpu)}
    )

    with caplog.at_level("INFO"):
        docker_evaluation._log_quant_eval_gpus(None, processed)

    assert [r.levelno for r in caplog.records if REPO in r.getMessage()] == [logging.ERROR]


def test_evaluate_repo_records_the_gpu_on_success_and_scored_fail():
    def evaluate(engine_factory) -> dict:
        return quant.evaluate_repo(
            repo=REPO,
            examples=EXAMPLES,
            engine_factory=engine_factory,
            layer_check=_layer_check(),
            quant_spec=W4A16,
            config_check=SAME_CONFIG,
            config_loader=lambda _repo: W4A16_CONFIG,
            eval_gpu=H100_SXM,
        )

    success = evaluate(lambda _repo: FakeEngine())
    scored_fail = evaluate(_raise_on_load)

    for result in (success, scored_fail):
        assert result["eval_gpu"] == H100_SXM.model_dump()
        assert EvaluationResultText.model_validate(result).eval_gpu == H100_SXM


def test_probe_gpu_reads_the_last_stdout_line(monkeypatch):
    stdout = f"some torch warning\n{H100_SXM.model_dump_json()}\n"
    monkeypatch.setattr(quant.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=stdout))

    assert quant.probe_gpu() == H100_SXM


def test_quant_gpu_reads_device_properties(monkeypatch):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        get_device_properties=lambda index: SimpleNamespace(name=f"NVIDIA H100 PCIe:{index}", multi_processor_count=114)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.delitem(sys.modules, "validator.evaluation.quant_gpu", raising=False)
    from validator.evaluation import quant_gpu

    assert quant_gpu.read_gpu() == QuantEvalGpu(name="NVIDIA H100 PCIe:0", sm_count=114)
    monkeypatch.delitem(sys.modules, "validator.evaluation.quant_gpu")


REFERENCE_MODEL = "org/base"
SXM_CHECKED = H100_SXM.model_copy(update={"sku": "H100 SXM"})


def _calls_by_key(calls) -> list[tuple[str, str]]:
    """(repo, results key) per subprocess, in order."""
    args = [quant.QuantEvaluationArgs.model_validate_json(c[-1]) for c in calls]
    return [(a.repo, a.saved_as) for a in args]


def test_main_scores_the_reference_first_when_its_sku_is_not_cached(monkeypatch, quant_main_env):
    calls, _ = quant_main_env
    monkeypatch.setenv(docker_cst.QUANT_REFERENCE_MODEL_ENV, REFERENCE_MODEL)
    monkeypatch.setenv(docker_cst.QUANT_CACHED_REFERENCES_ENV, json.dumps({"H100 PCIe": 1.4}))
    monkeypatch.setattr(quant, "materialise_decoded_model", lambda _model, _urls, dest, _token: dest)

    quant.main()

    assert _calls_by_key(calls) == [
        (REFERENCE_MODEL, vcst.QUANT_REFERENCE_RESULT_KEY),
        (REPO, REPO),
        (OTHER_REPO, OTHER_REPO),
    ]
    reference_args = quant.QuantEvaluationArgs.model_validate_json(calls[0][-1])
    # bf16 references are never config- or kernel-checked, and never loaded from a decoded model.
    assert (reference_args.quant_spec, reference_args.quant_format, reference_args.model_path) == (None, None, None)
    assert reference_args.eval_gpu == SXM_CHECKED


def test_main_skips_the_reference_when_its_sku_is_cached(monkeypatch, quant_main_env):
    calls, _ = quant_main_env
    monkeypatch.setenv(docker_cst.QUANT_REFERENCE_MODEL_ENV, REFERENCE_MODEL)
    monkeypatch.setenv(docker_cst.QUANT_CACHED_REFERENCES_ENV, json.dumps({"H100 SXM": 1.4}))
    monkeypatch.setattr(quant, "materialise_decoded_model", lambda _model, _urls, dest, _token: dest)

    quant.main()

    assert [repo for repo, _ in _calls_by_key(calls)] == [REPO, OTHER_REPO]


def test_main_without_a_reference_model_scores_only_the_repos(monkeypatch, quant_main_env):
    calls, _ = quant_main_env
    monkeypatch.setattr(quant, "materialise_decoded_model", lambda _model, _urls, dest, _token: dest)

    quant.main()

    assert [repo for repo, _ in _calls_by_key(calls)] == [REPO, OTHER_REPO]


def test_main_scores_a_reference_that_is_one_of_the_repos_once(monkeypatch, quant_main_env):
    """The deployment whose repo is also the reference: its result is copied as the reference."""
    calls, results_path = quant_main_env
    monkeypatch.setenv(docker_cst.QUANT_REFERENCE_MODEL_ENV, OTHER_REPO)
    monkeypatch.setattr(quant, "materialise_decoded_model", lambda _model, _urls, dest, _token: dest)
    quant.save_results({OTHER_REPO: {"is_finetune": True, "eval_loss": 1.4}})

    quant.main()

    assert [repo for repo, _ in _calls_by_key(calls)] == [REPO, OTHER_REPO]
    saved = json.loads(results_path.read_text())
    assert saved[vcst.QUANT_REFERENCE_RESULT_KEY] == saved[OTHER_REPO]


def test_override_reference_key_never_matches_a_prod_sku():
    override = A100.model_copy(update={"override": True})
    assert override.reference_key == "override:NVIDIA A100-SXM4-80GB/108"
    assert SXM_CHECKED.reference_key == "H100 SXM"


def _reference_dict(loss=1.4, **overrides) -> dict:
    return {
        "is_finetune": True,
        "eval_loss": loss,
        "per_example_losses": [loss, loss],
        "eval_set_fingerprint": "fp",
        "eval_gpu": SXM_CHECKED.model_dump(),
        **overrides,
    }


async def _dispatch_with_reference(monkeypatch, raw_reference, reference_models=None, cached=None):
    reference_models = {REPO: REFERENCE_MODEL} if reference_models is None else reference_models
    captured = {}

    async def fake_load_eval_pair_state_for_models(*_args, **_kwargs):
        return {}, {REPO: "hk_a"}

    async def fake_run_remote_eval_repos(**kwargs):
        captured.update(kwargs)
        raw = {REPO: {"is_finetune": True, "eval_loss": 1.5, "eval_gpu": SXM_CHECKED.model_dump()}, "model_params_count": 7}
        if raw_reference is not None:
            raw[vcst.QUANT_REFERENCE_RESULT_KEY] = raw_reference
        return {REPO: raw}

    monkeypatch.setattr(docker_evaluation, "load_eval_pair_state_for_models", fake_load_eval_pair_state_for_models)
    monkeypatch.setattr(docker_evaluation, "run_remote_eval_repos", fake_run_remote_eval_repos)
    results = await docker_evaluation.run_evaluation_basilica_quant(
        dataset="https://bucket.example/test_data.jsonl",
        models=[REPO],
        original_model="org/base",
        dataset_type=InstructTextDatasetType(),
        file_format=FileFormat.S3,
        num_gpus=1,
        quant_spec=None,
        quant_format=None,
        task_id=uuid4(),
        psql_db=object(),
        reference_models=reference_models,
        cached_references=cached,
    )
    return captured, results


@pytest.mark.asyncio
async def test_dispatch_passes_the_reference_model_and_the_cached_skus(monkeypatch):
    captured, _ = await _dispatch_with_reference(monkeypatch, None, cached={REPO: {"H100 SXM": 1.4}})

    env = captured["build_env_for_repo"](REPO)
    assert env[docker_cst.QUANT_REFERENCE_MODEL_ENV] == REFERENCE_MODEL
    assert env["ORIGINAL_MODEL"] == REFERENCE_MODEL
    assert json.loads(env[docker_cst.QUANT_CACHED_REFERENCES_ENV]) == {"H100 SXM": 1.4}


@pytest.mark.asyncio
async def test_dispatch_sets_each_repos_own_reference_env(monkeypatch):
    captured, _ = await _dispatch_with_reference(
        monkeypatch,
        None,
        reference_models={REPO: "org/a-r1", OTHER_REPO: "org/b-r1"},
        cached={REPO: {"H100 SXM": 1.4}, OTHER_REPO: {"H100 SXM": 1.9, "H100 PCIe": 1.95}},
    )

    env_a, env_b = captured["build_env_for_repo"](REPO), captured["build_env_for_repo"](OTHER_REPO)
    assert (env_a[docker_cst.QUANT_REFERENCE_MODEL_ENV], env_a["ORIGINAL_MODEL"]) == ("org/a-r1", "org/a-r1")
    assert (env_b[docker_cst.QUANT_REFERENCE_MODEL_ENV], env_b["ORIGINAL_MODEL"]) == ("org/b-r1", "org/b-r1")
    assert json.loads(env_a[docker_cst.QUANT_CACHED_REFERENCES_ENV]) == {"H100 SXM": 1.4}
    assert json.loads(env_b[docker_cst.QUANT_CACHED_REFERENCES_ENV]) == {"H100 SXM": 1.9, "H100 PCIe": 1.95}
    assert env_a["MODELS"] == REPO and env_b["MODELS"] == OTHER_REPO


@pytest.mark.asyncio
async def test_dispatch_without_a_reference_model_sets_no_reference_env(monkeypatch):
    captured, _ = await _dispatch_with_reference(monkeypatch, None, reference_models={})

    env = captured["build_env_for_repo"](REPO)
    assert docker_cst.QUANT_REFERENCE_MODEL_ENV not in env
    assert docker_cst.QUANT_CACHED_REFERENCES_ENV not in env
    assert env["ORIGINAL_MODEL"] == "org/base"


@pytest.mark.asyncio
async def test_dispatch_returns_the_same_deployment_reference_with_the_result(monkeypatch):
    _, results = await _dispatch_with_reference(monkeypatch, _reference_dict())

    result = results.results[REPO]
    assert result.eval_loss == 1.5
    assert result.quant_reference.model == REFERENCE_MODEL
    assert (result.quant_reference.eval_loss, result.quant_reference.eval_gpu) == (1.4, SXM_CHECKED)
    assert set(results.results) == {REPO}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_reference",
    [
        "Traceback: CUDA error",
        {"is_finetune": True, "eval_loss": None, "quant_failure": "model failed to load in vLLM"},
        _reference_dict(per_example_losses=None),
        _reference_dict(eval_gpu=None),
    ],
    ids=["errored", "scored-fail", "no-vector", "no-gpu"],
)
async def test_dispatch_drops_a_reference_that_did_not_score_and_keeps_the_result(monkeypatch, caplog, raw_reference):
    with caplog.at_level("ERROR"):
        _, results = await _dispatch_with_reference(monkeypatch, raw_reference)

    assert results.results[REPO].eval_loss == 1.5
    assert results.results[REPO].quant_reference is None
    assert any("bf16 reference" in record.getMessage() for record in caplog.records)


def _config_loader_raising(error: BaseException):
    def load(_repo):
        raise error

    return load


def _evaluate_loading_config_with(error: BaseException) -> dict:
    return quant.evaluate_repo(
        repo=REPO,
        examples=EXAMPLES,
        engine_factory=lambda _repo: pytest.fail("loaded a repo whose config.json could not be read"),
        layer_check=_layer_check(),
        quant_spec=W4A16,
        config_check=SAME_CONFIG,
        config_loader=_config_loader_raising(error),
    )


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(),
        ConnectionResetError("peer reset"),
        urllib.error.HTTPError("https://hub", 503, "unavailable", None, None),
        urllib.error.HTTPError("https://hub", 429, "slow down", None, None),
        OSError("hub read failed"),
    ],
    ids=["timeout", "reset", "503", "429", "wrapped-timeout"],
)
def test_a_transport_error_reading_the_config_is_an_evaluation_error(error):
    if str(error) == "hub read failed":
        error.__cause__ = TimeoutError("read timed out")

    result = _evaluate_loading_config_with(error)

    assert set(result) == {"evaluation_error"}
    assert "quant_failure" not in result


@pytest.mark.parametrize(
    "error", [ValueError("config.json is over the cap"), json.JSONDecodeError("bad", "{", 1), OSError("not a file")]
)
def test_a_malformed_config_stays_a_scored_fail(error):
    result = _evaluate_loading_config_with(error)

    assert result["eval_loss"] is None
    assert result["quant_failure"].startswith("config.json unreadable: ")
