"""Stand-ins for vllm and humming so quant_vllm imports on CPU.

They mirror only what quant_vllm touches on vLLM 0.31.0: LinearBase modules carrying a quant_method
and, when quantised, a compressed-tensors scheme with quant_type.size_bits, group_size, symmetric
and the chosen kernel; the LLM constructor's keyword arguments; and transformers_utils.config.get_config.
"""

from types import ModuleType
from types import SimpleNamespace


class LinearBase:
    def __init__(self, quant_method: str, scheme=None):
        self.quant_method = type(quant_method, (), {})()
        if scheme is not None:
            self.scheme = scheme


class ParallelLMHead:
    """An embedding in vLLM, not a LinearBase."""

    quant_method = type("UnquantizedEmbeddingMethod", (), {})()


def scheme(name: str, kernel: str, bits: int, group_size: int, symmetric: bool):
    fake = type(name, (), {})()
    fake.kernel = type(kernel, (), {})()
    fake.quant_type = SimpleNamespace(size_bits=bits)
    fake.group_size = group_size
    fake.symmetric = symmetric
    return fake


def unquantised() -> LinearBase:
    return LinearBase("UnquantizedLinearMethod")


def quantised(**scheme_args) -> LinearBase:
    return LinearBase("CompressedTensorsLinearMethod", scheme(**scheme_args))


class FakeModel:
    def __init__(self, modules: dict):
        self.modules = modules

    def named_modules(self):
        return list(self.modules.items())


class FakeLLM:
    """Only the public apply_model: one in-process worker. Built like vllm.LLM, it records its kwargs."""

    def __init__(self, modules: dict | None = None, **kwargs):
        self.model = FakeModel(modules or {})
        self.kwargs = kwargs

    def apply_model(self, func):
        return [func(self.model)]


class FakeHfConfig:
    def __init__(self, config: dict):
        self.config = config

    def to_dict(self) -> dict:
        return dict(self.config)


def unconfigured_get_config(model, **_kwargs):
    raise AssertionError(f"get_config({model!r}) called; set fake_vllm configs for this test")


def fake_modules(heuristics_config, get_config=unconfigured_get_config) -> dict[str, ModuleType]:
    """Fake vllm and humming packages keyed for sys.modules; heuristics_config stands in for
    humming.tune._get_heuristics_config and get_config for vllm.transformers_utils.config.get_config."""
    vllm = ModuleType("vllm")
    vllm.LLM = FakeLLM
    vllm.SamplingParams = SimpleNamespace
    inputs = ModuleType("vllm.inputs")
    inputs.TokensPrompt = SimpleNamespace
    linear = ModuleType("vllm.model_executor.layers.linear")
    linear.LinearBase = LinearBase
    transformers_utils = ModuleType("vllm.transformers_utils")
    transformers_config = ModuleType("vllm.transformers_utils.config")
    transformers_config.get_config = get_config
    humming = ModuleType("humming")
    tune = ModuleType("humming.tune")
    tune._get_heuristics_config = heuristics_config
    humming.tune = tune
    return {
        "vllm": vllm,
        "vllm.inputs": inputs,
        "vllm.model_executor": ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.linear": linear,
        "vllm.transformers_utils": transformers_utils,
        "vllm.transformers_utils.config": transformers_config,
        "humming": humming,
        "humming.tune": tune,
    }
