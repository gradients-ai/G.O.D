"""Integrity of the axolotl 0bda5a13 golden fixtures, and the two chat-masking defects they freeze.

The fixtures are what today's eval (`instruct_text._load_evaluation_dataset` on axolotl) produced on real tokenizers; the
axolotl-free implementation is proven against them. These tests need no transformers: they check each fixture is
internally consistent (so a hand edit or a partial recapture cannot slip in) and that the fixtures demonstrate the defects
the replacement exists to fix, which makes the fixtures self-describing.

The models are re-declared here rather than imported from the capture script, which imports axolotl at module level.
"""

import json
from math import ceil
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel
from pydantic import ConfigDict


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "axolotl_golden"

AXOLOTL_COMMIT = "0bda5a13"
TRANSFORMERS_VERSION = "5.12.1"
EVAL_SEQUENCE_LEN = 4096

QWEN_IM_START = 151644
QWEN_IM_END = 151645

INSTRUCT_KINDS = [
    "instruct",
    "instruct_seq256",
    "instruct_system",
    "instruct_system_nofield",
    "instruct_system_prompt",
    "instruct_custom_format",
    "completion",
    "completion_seq256",
]
CHAT_KINDS = ["chat", "chat_seq256", "chat_sharegpt"]
CHAT_TOKENIZERS = [
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/gemma-2-2b-it",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "ibm-granite/granite-4.1-3b",
    "LiquidAI/LFM2.5-2.6B",
    "kakaocorp/kanana-2-3b-instruct",
    "Nanbeige/Nanbeige4.1-3B",
]
INSTRUCT_ONLY_TOKENIZERS = ["unsloth/Llama-3.2-1B", "unsloth/gemma-2-2b", "tiiuae/Falcon-H1-0.5B-Base", "microsoft/phi-2"]
CHATML_TOKENIZERS = ["Qwen/Qwen2.5-0.5B", "unsloth/Llama-3.2-1B-Instruct"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldenRawRow(_Strict):
    row_index: int
    chunk: int = 0
    input_ids: list[int]
    labels: list[int]
    dropped: Literal["long", "short", "unscored", "empty"] | None = None


class GoldenFinalRow(_Strict):
    row_index: int
    chunk: int = 0
    input_ids: list[int]
    labels: list[int]


class GoldenExcludedRow(_Strict):
    row_index: int
    error_type: str
    error_message: str


class GoldenFixture(_Strict):
    axolotl_commit: str
    transformers_version: str
    datasets_version: str
    tokenizer_repo: str
    tokenizer_sha: str
    tokenizer_files_sha256: dict[str, str]
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    kind: str
    dataset_type_name: Literal["InstructTextDatasetType", "ChatTemplateDatasetType"]
    dataset_type: dict
    sequence_len: int
    max_position_embeddings: int
    rows_file: str
    excluded_rows: list[GoldenExcludedRow] = []
    raw: list[GoldenRawRow]
    final: list[GoldenFinalRow]


def _slug(repo: str) -> str:
    return repo.replace("/", "__")


def _expected_fixture_names() -> set[str]:
    names = set()
    for repo in CHAT_TOKENIZERS + INSTRUCT_ONLY_TOKENIZERS:
        kinds = INSTRUCT_KINDS + (CHAT_KINDS if repo in CHAT_TOKENIZERS else [])
        kinds += ["chat_chatml"] if repo in CHATML_TOKENIZERS else []
        names.update(f"{_slug(repo)}__{kind}.json" for kind in kinds)
    return names


FIXTURE_PATHS = sorted(FIXTURE_DIR.glob("*.json"))


def _load(path: Path) -> GoldenFixture:
    return GoldenFixture.model_validate_json(path.read_text(encoding="utf-8"))


def _fixture(repo: str, kind: str) -> GoldenFixture:
    return _load(FIXTURE_DIR / f"{_slug(repo)}__{kind}.json")


def _scored(row: GoldenRawRow | GoldenFinalRow) -> list[int]:
    return [token for token, label in zip(row.input_ids, row.labels) if label != -100]


def _raw_row(fixture: GoldenFixture, row_index: int, chunk: int = 0) -> GoldenRawRow:
    return next(row for row in fixture.raw if row.row_index == row_index and row.chunk == chunk)


def test_fixture_set_is_complete():
    assert {path.name for path in FIXTURE_PATHS} == _expected_fixture_names()


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=lambda path: path.stem)
def test_fixture_provenance_and_shape(path: Path):
    fixture = _load(path)
    assert path.name == f"{_slug(fixture.tokenizer_repo)}__{fixture.kind}.json"
    assert fixture.axolotl_commit == AXOLOTL_COMMIT
    assert fixture.transformers_version == TRANSFORMERS_VERSION
    assert len(fixture.tokenizer_sha) == 40
    assert "tokenizer_config.json" in fixture.tokenizer_files_sha256

    # The eval-config halving rule (common._load_and_update_evaluation_config) produced this sequence_len.
    expected_len = EVAL_SEQUENCE_LEN
    if fixture.max_position_embeddings < 2 * EVAL_SEQUENCE_LEN:
        expected_len = ceil(fixture.max_position_embeddings / 2)
    assert fixture.sequence_len == expected_len

    rows = json.loads((FIXTURE_DIR / fixture.rows_file).read_text(encoding="utf-8"))
    excluded = {row.row_index for row in fixture.excluded_rows}
    assert excluded <= set(range(len(rows)))
    assert {row.row_index for row in fixture.raw} == set(range(len(rows))) - excluded
    assert len({(row.row_index, row.chunk) for row in fixture.raw}) == len(fixture.raw)


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=lambda path: path.stem)
def test_labels_are_masked_or_copied(path: Path):
    fixture = _load(path)
    for row in [*fixture.raw, *fixture.final]:
        assert len(row.labels) == len(row.input_ids)
        assert all(label in (-100, token) for token, label in zip(row.input_ids, row.labels))


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=lambda path: path.stem)
def test_drop_reasons_match_lengths_and_labels(path: Path):
    fixture = _load(path)
    for row in fixture.raw:
        length = len(row.input_ids)
        if row.dropped == "empty":
            assert length == 0
        elif row.dropped == "short":
            assert length < 2
        elif row.dropped == "long":
            assert length > fixture.sequence_len
        elif row.dropped == "unscored":
            assert 2 <= length <= fixture.sequence_len
            assert not _scored(row)
        else:
            assert 2 <= length <= fixture.sequence_len
            assert _scored(row)


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=lambda path: path.stem)
def test_final_is_kept_raw_rows_stably_sorted_by_length(path: Path):
    fixture = _load(path)
    kept = [row for row in fixture.raw if row.dropped is None]
    expected = sorted(kept, key=lambda row: len(row.input_ids))
    assert [(row.row_index, row.chunk, row.input_ids, row.labels) for row in fixture.final] == [
        (row.row_index, row.chunk, row.input_ids, row.labels) for row in expected
    ]
    assert fixture.final


@pytest.mark.parametrize("repo", CHAT_TOKENIZERS + INSTRUCT_ONLY_TOKENIZERS)
def test_system_column_is_read_even_without_field_system(repo: str):
    """axolotl's dataclass default field_system="system" applies when G.O.D drops field_system=None."""
    with_field = _fixture(repo, "instruct_system")
    without_field = _fixture(repo, "instruct_system_nofield")
    assert [row.model_dump() for row in without_field.raw] == [row.model_dump() for row in with_field.raw]


def test_defect_qwen3_earlier_assistant_turn_is_shifted():
    """Defect 2: the empty <think> block is rendered only on the last assistant turn, but axolotl renders every turn
    as if last, so turn 1 ("Paris is the capital.") scores the 4 tokens after its content instead of its content."""
    fixture = _fixture("Qwen/Qwen3-0.6B-Base", "chat")
    row = _raw_row(fixture, row_index=1)
    period, newline, user, berlin = 13, 198, 872, 94409
    assert _scored(row) == [period, QWEN_IM_END, newline, QWEN_IM_START, user, berlin, period]


def test_defect_qwen25_base_never_scores_end_of_turn():
    """Defect 1: eos is <|endoftext|>, which the template never emits, so <|im_end|> is never a label."""
    fixture = _fixture("Qwen/Qwen2.5-0.5B", "chat")
    assert any(_scored(row) for row in fixture.raw)
    assert all(QWEN_IM_END not in _scored(row) for row in fixture.raw)


def test_qwen25_instruct_scores_end_of_turn_because_eos_is_im_end():
    """Contrast for defect 1: the same template on the instruct tokenizer (eos <|im_end|>) does score the turn end."""
    fixture = _fixture("Qwen/Qwen2.5-0.5B-Instruct", "chat")
    two_turn = _raw_row(fixture, row_index=1)
    assert _scored(two_turn).count(QWEN_IM_END) == 2


@pytest.mark.parametrize("repo", CHAT_TOKENIZERS + INSTRUCT_ONLY_TOKENIZERS)
def test_null_system_does_not_fall_back_to_system_prompt(repo: str):
    """axolotl picks system_prompt only when the system column is absent from the dataset, not when a row's value is null:
    pyarrow gives every row the column, so a null system renders exactly like an empty one (rows 1 and 2 differ only there)."""
    fixture = _fixture(repo, "instruct_system_prompt")
    empty_system, null_system, missing_key = (_raw_row(fixture, index) for index in (1, 2, 6))
    assert null_system.input_ids == empty_system.input_ids
    assert null_system.labels == empty_system.labels
    with_system = _raw_row(fixture, 0)
    assert len(with_system.input_ids) > len(empty_system.input_ids)
    assert missing_key.labels != [-100] * len(missing_key.labels)
