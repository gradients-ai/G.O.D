"""Capture what the axolotl-based text eval produces today, so the axolotl-free replacement can be proven against it.

Run once, under the golden venv (axolotl 0bda5a13 + transformers 5.12.1, see README.md), never by pytest:

    cd <worktree> && HF_HUB_OFFLINE=1 PYTHONPATH=. <axgold>/bin/python -m tests.validator.evaluation.golden.capture_axolotl_golden

For every (tokenizer, case) it builds the eval config through G.O.D's real `_load_and_update_evaluation_config` and calls
`instruct_text._load_evaluation_dataset` - the exact code path the eval image runs - and records its output as `final`.
It also records axolotl's per-row output *before* any length or all-masked filtering (`raw`), so that every row missing
from `final` is attributable to one drop rule. The capture refuses to write a fixture unless re-applying those drop rules
to `raw` reproduces `final` exactly; a fixture is therefore self-consistent by construction.

Templates that raise on some rows (gemma refuses a system turn, strict-alternation templates refuse an assistant-first
conversation) would fail the whole axolotl load. Those rows are probed one at a time, recorded with the exception in
`excluded_rows`, and the remaining rows are captured with their original `row_index`.
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import axolotl
import datasets
import transformers
from axolotl.utils.data.sft import _load_and_process_single_dataset
from axolotl.utils.data.shared import datasets_with_name_generator
from huggingface_hub import try_to_load_from_cache
from pydantic import BaseModel
from pydantic import Field
from transformers import AutoTokenizer

from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from validator.evaluation.common import _load_and_update_evaluation_config
from validator.evaluation.evaluators.instruct_text import _load_evaluation_dataset
from validator.evaluation.models import EvaluationArgs


EXPECTED_AXOLOTL_COMMIT = "0bda5a13"
EXPECTED_TRANSFORMERS_VERSION = "5.12.1"

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = REPO_ROOT / "tests" / "validator" / "evaluation" / "fixtures" / "axolotl_golden"
DATASET_DIR = FIXTURE_DIR / "datasets"
VALI_CONFIG = REPO_ROOT / "validator" / "assets" / "test_axolotl.yml"

# max_position_embeddings values fed to the eval-config halving rule: 40960 keeps sequence_len 4096, 512 halves it to 256.
LONG_CONTEXT = 40960
SHORT_CONTEXT = 512

# "empty": the strategy emitted no row at all (completion text that is empty/None yields zero chunks).
DropReason = Literal["long", "short", "unscored", "empty"]
FixtureKind = Literal[
    "instruct",
    "instruct_seq256",
    "instruct_system",
    "instruct_system_nofield",
    "instruct_system_prompt",
    "instruct_custom_format",
    "completion",
    "completion_seq256",
    "chat",
    "chat_seq256",
    "chat_sharegpt",
    "chat_chatml",
]


class GoldenRawRow(BaseModel):
    """axolotl's tokenised output for one source row (or one completion chunk) before any filtering."""

    row_index: int
    chunk: int = 0
    input_ids: list[int]
    labels: list[int]
    dropped: DropReason | None = None


class GoldenFinalRow(BaseModel):
    """One row of `_load_evaluation_dataset`'s output, in its final (filtered, length-sorted) order."""

    row_index: int
    chunk: int = 0
    input_ids: list[int]
    labels: list[int]


class GoldenExcludedRow(BaseModel):
    """A source row whose rendering raises inside axolotl; kept out of the captured load and recorded as a fact."""

    row_index: int
    error_type: str
    error_message: str


class GoldenFixture(BaseModel):
    axolotl_commit: str
    transformers_version: str
    datasets_version: str
    tokenizer_repo: str
    tokenizer_sha: str
    tokenizer_files_sha256: dict[str, str]
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    kind: FixtureKind
    dataset_type_name: Literal["InstructTextDatasetType", "ChatTemplateDatasetType"]
    dataset_type: dict
    sequence_len: int
    max_position_embeddings: int
    rows_file: str
    excluded_rows: list[GoldenExcludedRow] = Field(default_factory=list)
    raw: list[GoldenRawRow]
    final: list[GoldenFinalRow]


class TokenizerSpec(BaseModel):
    repo: str
    chat: bool
    chatml: bool = False


class CaptureCase(BaseModel):
    kind: FixtureKind
    rows_file: str
    dataset_type: InstructTextDatasetType | ChatTemplateDatasetType
    max_position_embeddings: int
    chat: bool


TOKENIZERS = [
    TokenizerSpec(repo="Qwen/Qwen3-0.6B-Base", chat=True),
    TokenizerSpec(repo="Qwen/Qwen2.5-0.5B", chat=True, chatml=True),
    TokenizerSpec(repo="Qwen/Qwen2.5-0.5B-Instruct", chat=True),
    TokenizerSpec(repo="unsloth/Llama-3.2-1B-Instruct", chat=True, chatml=True),
    TokenizerSpec(repo="unsloth/Llama-3.2-1B", chat=False),
    TokenizerSpec(repo="unsloth/mistral-7b-instruct-v0.3", chat=True),
    TokenizerSpec(repo="unsloth/gemma-2-2b-it", chat=True),
    TokenizerSpec(repo="unsloth/gemma-2-2b", chat=False),
    TokenizerSpec(repo="TinyLlama/TinyLlama-1.1B-Chat-v1.0", chat=True),
    TokenizerSpec(repo="ibm-granite/granite-4.1-3b", chat=True),
    TokenizerSpec(repo="LiquidAI/LFM2.5-2.6B", chat=True),
    TokenizerSpec(repo="kakaocorp/kanana-2-3b-instruct", chat=True),
    TokenizerSpec(repo="Nanbeige/Nanbeige4.1-3B", chat=True),
    TokenizerSpec(repo="tiiuae/Falcon-H1-0.5B-Base", chat=False),
    TokenizerSpec(repo="microsoft/phi-2", chat=False),
]

_INSTRUCT = InstructTextDatasetType(field_instruction="instruct", field_input="input", field_output="output")
_CHAT_DEFAULT = ChatTemplateDatasetType(
    chat_template="tokenizer_default", chat_column="conversations", chat_role_field="from", chat_content_field="value"
)

INSTRUCT_CASES = [
    CaptureCase(
        kind="instruct", rows_file="instruct_rows.json", dataset_type=_INSTRUCT, max_position_embeddings=LONG_CONTEXT, chat=False
    ),
    CaptureCase(
        kind="instruct_seq256",
        rows_file="instruct_rows.json",
        dataset_type=_INSTRUCT,
        max_position_embeddings=SHORT_CONTEXT,
        chat=False,
    ),
    CaptureCase(
        kind="instruct_system",
        rows_file="instruct_system_rows.json",
        dataset_type=InstructTextDatasetType(
            field_instruction="instruct", field_input="input", field_output="output", field_system="system"
        ),
        max_position_embeddings=LONG_CONTEXT,
        chat=False,
    ),
    # field_system=None: G.O.D drops the None, axolotl's dataclass default "system" still reads the column.
    CaptureCase(
        kind="instruct_system_nofield",
        rows_file="instruct_system_rows.json",
        dataset_type=_INSTRUCT,
        max_position_embeddings=LONG_CONTEXT,
        chat=False,
    ),
    CaptureCase(
        kind="instruct_system_prompt",
        rows_file="instruct_system_rows.json",
        dataset_type=InstructTextDatasetType(
            field_instruction="instruct",
            field_input="input",
            field_output="output",
            field_system="system",
            system_prompt="Default system prompt.",
            system_format="[SYS] {system} [/SYS] ",
        ),
        max_position_embeddings=LONG_CONTEXT,
        chat=False,
    ),
    CaptureCase(
        kind="instruct_custom_format",
        rows_file="instruct_rows.json",
        dataset_type=InstructTextDatasetType(
            field_instruction="instruct",
            field_input="input",
            field_output="output",
            format="Q: {instruction}\nI: {input}\nA:",
            no_input_format="Q: {instruction}\nA:",
        ),
        max_position_embeddings=LONG_CONTEXT,
        chat=False,
    ),
    CaptureCase(
        kind="completion",
        rows_file="completion_rows.json",
        dataset_type=InstructTextDatasetType(field_instruction="instruct"),
        max_position_embeddings=LONG_CONTEXT,
        chat=False,
    ),
    CaptureCase(
        kind="completion_seq256",
        rows_file="completion_rows.json",
        dataset_type=InstructTextDatasetType(field_instruction="instruct"),
        max_position_embeddings=SHORT_CONTEXT,
        chat=False,
    ),
]

CHAT_CASES = [
    CaptureCase(
        kind="chat", rows_file="chat_rows.json", dataset_type=_CHAT_DEFAULT, max_position_embeddings=LONG_CONTEXT, chat=True
    ),
    CaptureCase(
        kind="chat_seq256",
        rows_file="chat_rows.json",
        dataset_type=_CHAT_DEFAULT,
        max_position_embeddings=SHORT_CONTEXT,
        chat=True,
    ),
    CaptureCase(
        kind="chat_sharegpt",
        rows_file="chat_sharegpt_rows.json",
        dataset_type=_CHAT_DEFAULT.model_copy(update={"chat_user_reference": "human", "chat_assistant_reference": "gpt"}),
        max_position_embeddings=LONG_CONTEXT,
        chat=True,
    ),
]

CHATML_CASE = CaptureCase(
    kind="chat_chatml",
    rows_file="chat_rows.json",
    dataset_type=_CHAT_DEFAULT.model_copy(update={"chat_template": "chatml"}),
    max_position_embeddings=LONG_CONTEXT,
    chat=True,
)


class _FakeModelConfig:
    def __init__(self, max_position_embeddings: int):
        self.max_position_embeddings = max_position_embeddings


class _FakeModel:
    """Only `.config.max_position_embeddings` is read by `_load_and_update_evaluation_config`."""

    def __init__(self, max_position_embeddings: int):
        self.config = _FakeModelConfig(max_position_embeddings)


def tokenizer_slug(repo: str) -> str:
    return repo.replace("/", "__")


def fixture_name(repo: str, kind: str) -> str:
    return f"{tokenizer_slug(repo)}__{kind}.json"


def cases_for(spec: TokenizerSpec) -> list[CaptureCase]:
    cases = list(INSTRUCT_CASES)
    if spec.chat:
        cases += CHAT_CASES
    if spec.chatml:
        cases.append(CHATML_CASE)
    return cases


def axolotl_commit() -> str:
    source_root = Path(axolotl.__file__).resolve().parents[2]
    return subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "--short=8", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def load_eval_tokenizer(repo: str) -> AutoTokenizer:
    """Same post-load fix-up as instruct_text's evaluator entry point (pad falls back to eos)."""
    tokenizer = AutoTokenizer.from_pretrained(repo)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def tokenizer_provenance(repo: str) -> tuple[str, dict[str, str]]:
    # The snapshot the tokenizer was actually loaded from (the cached refs/main), even when the snapshot holds only
    # tokenizer files: snapshot_download(local_files_only=True) refuses such partial snapshots.
    snapshot = Path(try_to_load_from_cache(repo, "tokenizer_config.json")).parent
    hashes = {}
    for path in sorted(snapshot.iterdir()):
        if path.name == "config.json" or path.suffix == ".safetensors" or not path.is_file():
            continue
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot.name, hashes


def build_config(case: CaptureCase, dataset_path: Path, tokenizer: AutoTokenizer, output_dir: str):
    args = EvaluationArgs(
        dataset=str(dataset_path),
        original_model=tokenizer.name_or_path,
        dataset_type=case.dataset_type,
        file_format=FileFormat.JSON,
        repo="golden-capture",
    )
    cfg = _load_and_update_evaluation_config(args, _FakeModel(case.max_position_embeddings), str(VALI_CONFIG))
    cfg.output_dir = output_dir
    cfg.tokenizer_config = tokenizer.name_or_path
    # In prod this is None (single process); 1 is equivalent and keeps datasets from spawning workers.
    cfg.dataset_num_proc = 1
    return cfg


def axolotl_unfiltered_rows(cfg, tokenizer: AutoTokenizer) -> list[dict]:
    """axolotl's tokenised rows before `handle_long_seq_in_dataset` and G.O.D's all-masked filter."""
    dataset_config = next(iter(datasets_with_name_generator(cfg.datasets)))
    wrapper, _ = _load_and_process_single_dataset(
        dataset_config=dataset_config, cfg=cfg, tokenizer=tokenizer, split="train", seed=cfg.seed
    )
    rows = []
    for row in wrapper:
        input_ids = list(row["input_ids"])
        if list(row["attention_mask"]) != [1] * len(input_ids):
            raise AssertionError("attention_mask is not all ones; the fixture format would lose information")
        rows.append({"input_ids": input_ids, "labels": list(row["labels"])})
    return rows


def completion_chunk_counts(cfg, tokenizer: AutoTokenizer, source_rows: list[dict]) -> list[int]:
    """Completion rows split into sequence_len chunks; count chunks per source row so each chunk keeps its row_index."""
    dataset_config = next(iter(datasets_with_name_generator(cfg.datasets)))
    wrapper, _ = _load_and_process_single_dataset(
        dataset_config=dataset_config, cfg=cfg, tokenizer=tokenizer, split="train", seed=cfg.seed
    )
    field = dataset_config.field
    counts = []
    for row in source_rows:
        out = wrapper.prompt_tokenizer.tokenize_prompt({field: [row.get(field)]})
        counts.append(len(out.get("input_ids", [])))
    return counts


def drop_reason(input_ids: list[int], labels: list[int], sequence_len: int, min_sequence_len: int) -> DropReason | None:
    if len(input_ids) < min_sequence_len:
        return "short"
    if len(input_ids) > sequence_len:
        return "long"
    if all(label == -100 for label in labels):
        return "unscored"
    return None


def find_raising_rows(case: CaptureCase, rows: list[dict], tokenizer: AutoTokenizer, workdir: Path) -> list[GoldenExcludedRow]:
    excluded = []
    for index, row in enumerate(rows):
        probe = workdir / f"probe_{index}" / "rows.json"
        probe.parent.mkdir(parents=True)
        probe.write_text(json.dumps([row], ensure_ascii=False), encoding="utf-8")
        cfg = build_config(case, probe, tokenizer, str(probe.parent / "out"))
        try:
            axolotl_unfiltered_rows(cfg, tokenizer)
        except Exception as exc:  # noqa: BLE001 - recording the template's own exception is the point
            excluded.append(GoldenExcludedRow(row_index=index, error_type=type(exc).__name__, error_message=str(exc)[:500]))
    return excluded


def capture_case(spec: TokenizerSpec, case: CaptureCase, tokenizer: AutoTokenizer, provenance: tuple[str, dict]) -> GoldenFixture:
    source_rows = json.loads((DATASET_DIR / case.rows_file).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="golden_") as tmp:
        workdir = Path(tmp)
        excluded: list[GoldenExcludedRow] = []
        kept_indices = list(range(len(source_rows)))
        rows_path = DATASET_DIR / case.rows_file
        cfg = build_config(case, rows_path, tokenizer, str(workdir / "out"))
        try:
            final_rows = _load_evaluation_dataset(cfg, tokenizer)
        except Exception:
            excluded = find_raising_rows(case, source_rows, tokenizer, workdir / "probes")
            if not excluded:
                raise
            excluded_set = {row.row_index for row in excluded}
            kept_indices = [index for index in kept_indices if index not in excluded_set]
            rows_path = workdir / "subset" / case.rows_file
            rows_path.parent.mkdir()
            rows_path.write_text(json.dumps([source_rows[i] for i in kept_indices], ensure_ascii=False), encoding="utf-8")
            cfg = build_config(case, rows_path, tokenizer, str(workdir / "out_subset"))
            final_rows = _load_evaluation_dataset(cfg, tokenizer)

        raw_cfg = build_config(case, rows_path, tokenizer, str(workdir / "out_raw"))
        unfiltered = axolotl_unfiltered_rows(raw_cfg, tokenizer)
        if case.kind.startswith("completion"):
            counts = completion_chunk_counts(raw_cfg, tokenizer, [source_rows[i] for i in kept_indices])
        else:
            counts = [1] * len(kept_indices)
        if sum(counts) != len(unfiltered):
            raise AssertionError(f"{case.kind}: {len(unfiltered)} tokenised rows for {sum(counts)} expected")

        sequence_len = int(cfg.sequence_len)
        min_sequence_len = int(cfg.min_sample_len or 2)
        raw: list[GoldenRawRow] = []
        cursor = 0
        for source_index, count in zip(kept_indices, counts):
            if count == 0:
                raw.append(GoldenRawRow(row_index=source_index, input_ids=[], labels=[], dropped="empty"))
            for chunk in range(count):
                row = unfiltered[cursor]
                cursor += 1
                raw.append(
                    GoldenRawRow(
                        row_index=source_index,
                        chunk=chunk,
                        input_ids=row["input_ids"],
                        labels=row["labels"],
                        dropped=drop_reason(row["input_ids"], row["labels"], sequence_len, min_sequence_len),
                    )
                )

        expected_final = sorted((row for row in raw if row.dropped is None), key=lambda row: len(row.input_ids))
        actual = [(list(row["input_ids"]), list(row["labels"])) for row in final_rows]
        if actual != [(row.input_ids, row.labels) for row in expected_final]:
            raise AssertionError(f"{spec.repo} {case.kind}: drop rules applied to raw do not reproduce _load_evaluation_dataset")

    tokenizer_sha, file_hashes = provenance
    return GoldenFixture(
        axolotl_commit=axolotl_commit(),
        transformers_version=transformers.__version__,
        datasets_version=datasets.__version__,
        tokenizer_repo=spec.repo,
        tokenizer_sha=tokenizer_sha,
        tokenizer_files_sha256=file_hashes,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        kind=case.kind,
        dataset_type_name=type(case.dataset_type).__name__,
        dataset_type=case.dataset_type.model_dump(),
        sequence_len=sequence_len,
        max_position_embeddings=case.max_position_embeddings,
        rows_file=f"datasets/{case.rows_file}",
        excluded_rows=excluded,
        raw=raw,
        final=[
            GoldenFinalRow(row_index=row.row_index, chunk=row.chunk, input_ids=row.input_ids, labels=row.labels)
            for row in expected_final
        ],
    )


def write_fixture(fixture: GoldenFixture, path: Path) -> None:
    """One row per line: reviewable diffs without one token per line."""
    data = fixture.model_dump(mode="json")
    rows = {key: data.pop(key) for key in ("raw", "final")}
    header = json.dumps(data, ensure_ascii=False, indent=1)[:-2]
    parts = [header]
    for key, values in rows.items():
        body = ",\n".join("  " + json.dumps(value, ensure_ascii=False, separators=(",", ":")) for value in values)
        parts.append(f',\n "{key}": [\n{body}\n ]' if values else f',\n "{key}": []')
    path.write_text("".join(parts) + "\n}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", default=None, help="tokenizer repos to capture (default: all)")
    parser.add_argument("--kinds", nargs="*", default=None, help="fixture kinds to capture (default: all)")
    args = parser.parse_args()

    commit = axolotl_commit()
    if not commit.startswith(EXPECTED_AXOLOTL_COMMIT) or transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise SystemExit(f"wrong golden environment: axolotl {commit}, transformers {transformers.__version__}")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("set HF_HUB_OFFLINE=1 so tokenizers come from the pinned local snapshots")

    for spec in TOKENIZERS:
        if args.only and spec.repo not in args.only:
            continue
        tokenizer = load_eval_tokenizer(spec.repo)
        provenance = tokenizer_provenance(spec.repo)
        for case in cases_for(spec):
            if args.kinds and case.kind not in args.kinds:
                continue
            fixture = capture_case(spec, case, tokenizer, provenance)
            write_fixture(fixture, FIXTURE_DIR / fixture_name(spec.repo, case.kind))
            dropped = sum(1 for row in fixture.raw if row.dropped)
            print(
                f"{spec.repo:40s} {case.kind:24s} seq={fixture.sequence_len:<5d} raw={len(fixture.raw):<3d} "
                f"dropped={dropped:<3d} final={len(fixture.final):<3d} excluded={[r.row_index for r in fixture.excluded_rows]}"
            )


if __name__ == "__main__":
    main()
