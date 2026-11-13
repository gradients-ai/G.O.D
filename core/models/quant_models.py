from enum import Enum
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


QUANT_GROUP_SIZES = frozenset({32, 64, 128})


class QuantActivationFormat(str, Enum):
    BF16 = "bf16"
    INT8 = "int8"
    FP8 = "fp8"


class QuantFixedSpec(BaseModel):
    """The weight/activation format a fixed-track quant task requires every entrant to produce."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    weight_bits: int = Field(ge=2, le=8)
    activation: QuantActivationFormat
    group_size: int
    symmetric: bool
    scale_dtype: Literal["bfloat16"] = "bfloat16"
    kv_cache_dtype: Literal["bfloat16"] = "bfloat16"
    act_order: Literal[False] = False  # no g_idx / act-order reordering

    @field_validator("group_size")
    @classmethod
    def validate_group_size(cls, v: int) -> int:
        if v not in QUANT_GROUP_SIZES:
            raise ValueError(f"group_size must be one of {sorted(QUANT_GROUP_SIZES)}")
        return v

    @property
    def name(self) -> str:
        """Short label, e.g. W4A16-sym-g128 or W4A8fp8-sym-g128."""
        if self.activation == QuantActivationFormat.BF16:
            activation = "A16"
        else:
            activation = f"A8{self.activation.value}"
        symmetry = "sym" if self.symmetric else "asym"
        return f"W{self.weight_bits}{activation}-{symmetry}-g{self.group_size}"


class QuantEvalSku(BaseModel):
    """A GPU SKU quant evals may run on: torch's device name and SM count must both match."""

    label: str
    device_name: str
    sm_count: int
    runpod_gpu_type: str


class QuantEvalGpu(BaseModel):
    """The GPU a quant submission was scored on.

    name and sm_count are what the container read; sku is the allowed SKU label they matched, or None
    when QUANT_EVAL_ALLOW_ANY_GPU admitted an unlisted GPU (override True). Neither is set by the probe.
    """

    name: str
    sm_count: int
    sku: str | None = None
    override: bool = False

    @property
    def reference_key(self) -> str:
        """What the per-SKU bf16 reference cache is keyed by: the SKU label, or the raw GPU for an override,
        so a test-box result is normalised by a reference from the same unlisted GPU and never by a prod SKU's."""
        return self.sku if self.sku is not None else f"override:{self.name}/{self.sm_count}"


class QuantReferenceScore(BaseModel):
    """The bf16 reference model scored in the same deployment as a quant submission, so on the same GPU.

    Every quant loss on a task is compared as a multiple of this, measured on its own SKU.
    """

    model: str
    eval_loss: float
    per_example_losses: list[float]
    eval_set_fingerprint: str
    eval_gpu: QuantEvalGpu


class QuantFreeformPenalty(BaseModel):
    """How a freeform entry's normalised loss r = served loss / same-SKU bf16 reference prices its bytes.

    Effective size S = total_bytes x exp(curvature x max(0, r - knee)^2); lowest S wins. Above cap the
    entry is out (ranks after every under-cap entry, closest r first). knee and cap are ratios (1.04,
    1.08), unlike the task's pinned columns, which are the excess over the reference (0.04, 0.08) as
    bar_cap is. A cliff is knee == cap with curvature 0: P == 1 under the cap.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    knee: float = Field(gt=0, description="Normalised loss up to which bytes count unpenalised")
    curvature: float = Field(ge=0, description="Quadratic coefficient on the excess over the knee, per unit ratio squared")
    cap: float = Field(gt=0, description="Normalised loss above which an entry is out")

    @model_validator(mode="after")
    def validate_cap_at_or_above_knee(self) -> "QuantFreeformPenalty":
        if self.cap < self.knee:
            raise ValueError(f"cap {self.cap} must be at or above the knee {self.knee}")
        return self

    @classmethod
    def cliff(cls, cap: float) -> "QuantFreeformPenalty":
        """A step rule: in at or under cap with no penalty, out above it."""
        return cls(knee=cap, curvature=0.0, cap=cap)

    @property
    def is_cliff(self) -> bool:
        return self.curvature == 0.0

    def ln_penalty_under_cap(self, r: float) -> float:
        """ln P for an entry already known to be under the cap: curvature x max(0, r - knee)^2.

        The cap check itself is made on the absolute loss (test_loss <= cap x reference) by the ranking,
        so a row one rounding away from the cap is never priced as None here.
        """
        return self.curvature * max(0.0, r - self.knee) ** 2

    def ln_penalty(self, r: float) -> float | None:
        """ln P at normalised loss r; None above the cap (the entry is out)."""
        if r > self.cap:
            return None
        return self.ln_penalty_under_cap(r)


class QuantTarget(BaseModel):
    """The reference bar a miner's quant submission is judged against, published in its train request.

    A cross-runtime contract (validator -> trainer -> miner container as the QUANT_TARGET env var, never a flag):
    changing a field name or meaning breaks miner code. reference_loss is the miner's own bf16 model
    measured by the quant evaluator on the task's eval set. Fixed submissions are ranked by loss divided
    by it; freeform submissions are ranked by effective size, bytes x exp(penalty_curvature x
    max(0, loss / reference_loss - knee)^2), and are out above cap. bar_cap and bar duplicate the knee-based
    values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_model: str = Field(description="The bf16 repo the miner's loss is normalised by (its own starting model)")
    reference_loss: float = Field(gt=0, description="Eval loss of reference_model on the task's eval set, on reference_sku")
    reference_sku: str = Field(description="The GPU SKU label reference_loss was measured on")
    bar_cap: float | None = Field(None, ge=0, description="Freeform only: knee - 1, the unpenalised relative loss increase")
    bar: float | None = Field(None, description="Freeform only: knee_loss, (1 + bar_cap) * reference_loss")
    knee: float | None = Field(None, gt=0, description="Freeform only: normalised loss up to which bytes are unpenalised")
    knee_loss: float | None = Field(None, gt=0, description="Freeform only: knee * reference_loss")
    penalty_curvature: float | None = Field(None, ge=0, description="Freeform only: ln P = curvature * max(0, r - knee)**2")
    cap: float | None = Field(None, gt=0, description="Freeform only: normalised loss above which the entry is out")
    cap_loss: float | None = Field(None, gt=0, description="Freeform only: cap * reference_loss")

    @property
    def penalty(self) -> QuantFreeformPenalty | None:
        """The freeform penalty this target was published with; None on the fixed track."""
        if self.knee is None or self.penalty_curvature is None or self.cap is None:
            return None
        return QuantFreeformPenalty(knee=self.knee, curvature=self.penalty_curvature, cap=self.cap)


class QuantCalibrationReference(BaseModel):
    """The bf16 reference's loss on the calibration rows: validator -> trainer, inside the train request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str  # the bf16 repo scored (the miner's own starting model)
    loss: float  # flat mean over examples with a supervised token (quant_loss convention)
    per_example_losses: list[float | None]  # file order; None where an example has no scored token
    sku: str  # QuantEvalGpu.reference_key it was measured on


class QuantCalibrationRequest(BaseModel):
    """TrainRequestQuant.calibration: where the trainer fetches the calibration examples and what they must match."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    examples_url: str  # presigned GET of the tokenised JSONL; never logged
    eval_set_fingerprint: str  # quant_eval_set_fingerprint of those examples
    n_examples: int
    reference: QuantCalibrationReference | None  # None: not measured (prepass give-up, or a task-common task)


class QuantCalibrationExample(BaseModel):
    """One row of the file the miner reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_ids: list[int]
    labels: list[int]  # -100 masks; quant_loss convention (position 0 never scored)
    reference_loss: float | None  # the bf16 reference's loss on this example; None if not measured


class QuantCalibrationSet(BaseModel):
    """The JSON file at $QUANT_CALIBRATION_PATH."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    eval_set_fingerprint: str
    reference_model: str | None
    reference_loss: float | None
    reference_sku: str | None
    examples: list[QuantCalibrationExample]


class QuantFormat(str, Enum):
    W4A16_SYM_G128 = "W4A16-sym-g128"
    W4A8_FP8_G128 = "W4A8fp8-sym-g128"
    W3A16_ASYM_G128 = "W3A16-asym-g128"
    W2A16_ASYM_G64 = "W2A16-asym-g64"

    def spec(self) -> QuantFixedSpec:
        return _QUANT_FORMAT_SPECS[self]


_QUANT_FORMAT_SPECS: dict[QuantFormat, QuantFixedSpec] = {
    QuantFormat.W4A16_SYM_G128: QuantFixedSpec(
        weight_bits=4, activation=QuantActivationFormat.BF16, group_size=128, symmetric=True
    ),
    QuantFormat.W4A8_FP8_G128: QuantFixedSpec(
        weight_bits=4, activation=QuantActivationFormat.FP8, group_size=128, symmetric=True
    ),
    QuantFormat.W3A16_ASYM_G128: QuantFixedSpec(
        weight_bits=3, activation=QuantActivationFormat.BF16, group_size=128, symmetric=False
    ),
    QuantFormat.W2A16_ASYM_G64: QuantFixedSpec(
        weight_bits=2, activation=QuantActivationFormat.BF16, group_size=64, symmetric=False
    ),
}

# Fixed-track formats drawn for new tasks. The other QuantFormats remain valid specs.
LAUNCH_QUANT_FORMATS: tuple[QuantFormat, ...] = (
    QuantFormat.W3A16_ASYM_G128,
    QuantFormat.W2A16_ASYM_G64,
)


class QuantWorkload(BaseModel):
    """The work a quant job is priced for: token work, model-size work and host time, each with a size."""

    model_config = ConfigDict(frozen=True)

    eval_passes: int  # forward-only passes over the held-out split
    forward_tokens: int  # further forward-only tokens
    train_tokens: int  # forward+backward tokens
    train_max_epochs: float | None  # cap on train_tokens in epochs of the train split; None = uncapped
    model_seconds_per_b_params: float  # work that scales with model size: seconds per 1e9 params on the reference GPU
    host_seconds_per_b_params: float  # host / I/O work that scales with model size (load, save, upload)
    host_fixed_seconds: float  # host / I/O work that does not
    hours_slack: float  # headroom multiplier on the summed seconds


class QuantHoursBreakdown(BaseModel):
    """The three priced terms of a stats-based quant budget and the resulting hours."""

    token_seconds: float
    model_seconds: float
    host_seconds: float
    gpus: int
    train_tokens: int  # forward+backward tokens actually priced, after the epoch cap
    slack: float  # the workload's hours_slack applied to the summed seconds
    hours: float  # clamped, quarter-hour, boss multiplier applied
