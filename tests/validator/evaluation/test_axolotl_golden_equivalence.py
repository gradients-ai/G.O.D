"""The validator's own text-eval data path against what the axolotl-based eval produced (golden fixtures, axolotl 0bda5a13).

Instruct and completion: the final eval set (input ids, labels, order) and every per-row drop decision are identical.
Chat: input ids, row order and length drops are identical; labels differ only by the intended deltas of the chat rule
(`validator/evaluation/chat_labels.py`): every assistant turn's content is scored, plus exactly the template's turn end and,
on the last turn, the decoration after the generation prompt (Qwen3's empty think block). axolotl's shifted spans on
templates that decorate only the last turn are gone, and rows axolotl left all-masked may now be scored.

Needs transformers and the tokenizers in the local HF cache (skipped otherwise, e.g. in the dev venv). Run under the golden
venv: `HF_HUB_OFFLINE=1 <axgold>/bin/python -m pytest -q -p no:cacheprovider tests/validator/evaluation`.
"""

import json
from pathlib import Path

import pytest

from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import InstructTextDatasetType
from tests.validator.evaluation.test_axolotl_golden_fixtures import FIXTURE_DIR
from tests.validator.evaluation.test_axolotl_golden_fixtures import FIXTURE_PATHS
from tests.validator.evaluation.test_axolotl_golden_fixtures import QWEN_IM_END
from tests.validator.evaluation.test_axolotl_golden_fixtures import GoldenFixture
from validator.evaluation.chat_labels import SENTINEL
from validator.evaluation.chat_labels import ChatLabelled
from validator.evaluation.chat_labels import ChatLabelSkip
from validator.evaluation.chat_labels import chat_messages
from validator.evaluation.chat_labels import chat_role_map
from validator.evaluation.chat_labels import label_chat
from validator.evaluation.chat_templates import HfChatTokenizer
from validator.evaluation.chat_templates import resolve_chat_template
from validator.evaluation.instruct_labels import instruct_format
from validator.evaluation.instruct_labels import tokenize_completion_row
from validator.evaluation.instruct_labels import tokenize_instruct_row
from validator.evaluation.text_eval_config import text_eval_config_for_model
from validator.evaluation.text_eval_dataset import DropReason
from validator.evaluation.text_eval_dataset import HfTruncatingTokenizer
from validator.evaluation.text_eval_dataset import _drop_reason
from validator.evaluation.text_eval_dataset import build_text_eval_dataset
from validator.evaluation.text_eval_dataset import load_eval_rows
from validator.evaluation.text_examples import IGNORE_INDEX
from validator.evaluation.text_examples import TokenizedExample


transformers = pytest.importorskip("transformers")
hf_hub = pytest.importorskip("huggingface_hub")


# Templates that render a decoration only on the last assistant turn; axolotl rendered every turn as if last, so it
# scored the tokens after earlier turns' content instead of the content. Only these may lose a previously scored token.
SHIFTED_BY_AXOLOTL = {"Qwen/Qwen3-0.6B-Base", "Nanbeige/Nanbeige4.1-3B", "kakaocorp/kanana-2-3b-instruct"}
# Text the rule scores after an assistant turn's content: the template's own turn end, nothing else.
TURN_ENDS = {"<|im_end|>\n", "</s>\n", "</s>", "<end_of_turn>\n", "<|end_of_text|>\n", "<|eot_id|>"}
# Text scored before the content: the last-turn decoration after the generation prompt (Qwen3 / Nanbeige empty think).
LAST_TURN_DECORATIONS = {"<think>\n\n</think>\n\n"}

INSTRUCT_FIXTURES = [path for path in FIXTURE_PATHS if "__chat" not in path.name]
CHAT_FIXTURES = [path for path in FIXTURE_PATHS if "__chat" in path.name]


def _load(path: Path) -> GoldenFixture:
    return GoldenFixture.model_validate_json(path.read_text(encoding="utf-8"))


def _eval_tokenizer(fixture: GoldenFixture):
    """The fixture's tokenizer from the local cache, at the snapshot it was captured from, with the evaluator's pad fix-up."""
    cached = hf_hub.try_to_load_from_cache(fixture.tokenizer_repo, "tokenizer_config.json")
    if not isinstance(cached, str):
        pytest.skip(f"{fixture.tokenizer_repo} not in the local HF cache")
    if Path(cached).parent.name != fixture.tokenizer_sha:
        pytest.skip(f"{fixture.tokenizer_repo} cached at {Path(cached).parent.name}, fixture captured at {fixture.tokenizer_sha}")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(fixture.tokenizer_repo, local_files_only=True)
    except OSError as exc:
        pytest.skip(f"{fixture.tokenizer_repo} not loadable from the local HF cache: {exc}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (
        fixture.bos_token_id,
        fixture.eos_token_id,
        fixture.pad_token_id,
    )
    return tokenizer


def _source_rows(fixture: GoldenFixture) -> list[dict]:
    return json.loads((FIXTURE_DIR / fixture.rows_file).read_text(encoding="utf-8"))


def _kept_row_indices(fixture: GoldenFixture) -> list[int]:
    excluded = {row.row_index for row in fixture.excluded_rows}
    return [index for index in range(len(_source_rows(fixture))) if index not in excluded]


def _eval_rows(fixture: GoldenFixture, tmp_path: Path, row_indices: list[int]):
    """The rows through the evaluator's own loader (HF datasets json), from a file holding just these rows, as captured."""
    rows = _source_rows(fixture)
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([rows[index] for index in row_indices]), encoding="utf-8")
    return list(load_eval_rows(str(path)))


def _config(fixture: GoldenFixture):
    config = text_eval_config_for_model(fixture.max_position_embeddings)
    assert config.sequence_len == fixture.sequence_len
    return config


def _scored_positions(labels: list[int]) -> set[int]:
    return {position for position, label in enumerate(labels) if label != IGNORE_INDEX}


def _reason(reason: DropReason | None) -> str | None:
    return None if reason is None else reason.value


@pytest.mark.parametrize("path", INSTRUCT_FIXTURES, ids=lambda path: path.stem)
def test_instruct_and_completion_are_bit_identical(path: Path, tmp_path: Path):
    fixture = _load(path)
    tokenizer = _eval_tokenizer(fixture)
    dataset_type = InstructTextDatasetType.model_validate(fixture.dataset_type)
    config = _config(fixture)
    row_indices = _kept_row_indices(fixture)
    rows = _eval_rows(fixture, tmp_path, row_indices)

    # Per row, before any filtering: same tokens, same labels, same drop decision.
    fmt = instruct_format(dataset_type)
    adapter = HfTruncatingTokenizer(tokenizer)
    raw = []
    for row_index, row in zip(row_indices, rows):
        if fmt.is_completion:
            chunks = tokenize_completion_row(adapter, row, fmt, config.sequence_len)
            if not chunks:
                raw.append((row_index, 0, [], [], "empty"))
        else:
            chunks = [tokenize_instruct_row(adapter, row, fmt, config.sequence_len)]
        for chunk, example in enumerate(chunks):
            reason = _reason(_drop_reason(example, config))
            # The fixtures say "empty" only when the strategy emitted no row; a zero-length instruct row was dropped by
            # axolotl's min-length filter, recorded as "short". Same drop, different name.
            reason = "short" if reason == DropReason.EMPTY.value else reason
            raw.append((row_index, chunk, example.input_ids, example.labels, reason))
    assert raw == [(row.row_index, row.chunk, row.input_ids, row.labels, row.dropped) for row in fixture.raw]

    # The evaluator's whole build: the exact eval set, in the exact order.
    build = build_text_eval_dataset(rows, dataset_type, tokenizer, config)
    assert [(example.input_ids, example.labels) for example in build.examples] == [
        (row.input_ids, row.labels) for row in fixture.final
    ]
    assert all(example.attention_mask == [1] * len(example.input_ids) for example in build.examples)


def _chat_tokenizer(fixture: GoldenFixture, tokenizer) -> HfChatTokenizer:
    dataset_type = ChatTemplateDatasetType.model_validate(fixture.dataset_type)
    return HfChatTokenizer(tokenizer, resolve_chat_template(dataset_type.chat_template, tokenizer))


def _messages(fixture: GoldenFixture, row: dict):
    dataset_type = ChatTemplateDatasetType.model_validate(fixture.dataset_type)
    role_map = chat_role_map(dataset_type.chat_user_reference, dataset_type.chat_assistant_reference)
    return chat_messages(row, dataset_type.chat_column, dataset_type.chat_role_field, dataset_type.chat_content_field, role_map)


def _content_positions(chat_tokenizer: HfChatTokenizer, rendered: list[dict], turn_index: int, full_text: str, offsets):
    """Token positions overlapping assistant turn `turn_index`'s content in the full render (sentinel prefix/suffix diff)."""
    with_sentinel = rendered[:turn_index] + [{**rendered[turn_index], "content": SENTINEL}] + rendered[turn_index + 1 :]
    sentinel_text = chat_tokenizer.render(with_sentinel, add_generation_prompt=False)
    start = 0
    while start < min(len(full_text), len(sentinel_text)) and full_text[start] == sentinel_text[start]:
        start += 1
    suffix = 0
    while suffix < min(len(full_text), len(sentinel_text)) - start and full_text[-1 - suffix] == sentinel_text[-1 - suffix]:
        suffix += 1
    end = len(full_text) - suffix
    return start, end, {k for k, (s, e) in enumerate(offsets) if e > s and e > start and s < end}


@pytest.mark.parametrize("path", CHAT_FIXTURES, ids=lambda path: path.stem)
def test_chat_keeps_ids_rows_and_order_and_changes_labels_only_as_intended(path: Path, tmp_path: Path):
    fixture = _load(path)
    tokenizer = _eval_tokenizer(fixture)
    chat_tokenizer = _chat_tokenizer(fixture, tokenizer)
    config = _config(fixture)
    row_indices = _kept_row_indices(fixture)
    rows = _eval_rows(fixture, tmp_path, row_indices)
    old_raw = {row.row_index: row for row in fixture.raw}

    new_kept: list[tuple[int, ChatLabelled]] = []
    newly_scored_rows: list[int] = []
    lost_tokens: dict[int, str] = {}
    for row_index, row in zip(row_indices, rows):
        old = old_raw[row_index]
        messages = _messages(fixture, row)
        rendered = [message.template_dict() for message in messages]
        full_text = chat_tokenizer.render(rendered, add_generation_prompt=False)
        input_ids, offsets = chat_tokenizer.encode_with_offsets(full_text)
        # Input ids never change: the natural tokenisation of the full render, exactly what axolotl fed the model.
        assert input_ids == old.input_ids, f"row {row_index}: input ids differ"

        result = label_chat(chat_tokenizer, messages)
        if isinstance(result, ChatLabelSkip):
            assert result is ChatLabelSkip.EMPTY, f"row {row_index}: {result}"
            assert old.dropped in ("unscored", "long", "short"), f"row {row_index}: axolotl kept a row the rule cannot label"
            continue
        assert result.input_ids == input_ids
        reason = _reason(_drop_reason_for(result, config))
        if old.dropped in ("long", "short"):
            assert reason == old.dropped, f"row {row_index}: length drop differs"
            continue
        if old.dropped == "unscored":
            newly_scored_rows.append(row_index)
        else:
            assert reason is None, f"row {row_index}: axolotl kept it, the new path drops it as {reason}"
        new_kept.append((row_index, result))

        new_scored = _scored_positions(result.labels)
        old_scored = _scored_positions(old.labels)
        content: set[int] = set()
        for span in result.spans:
            content_start, content_end, turn_content = _content_positions(
                chat_tokenizer, rendered, span.turn_index, full_text, offsets
            )
            content |= turn_content
            span_start_char = offsets[span.start][0]
            span_end_char = offsets[span.end - 1][1]
            head = full_text[span_start_char:content_start] if span_start_char < content_start else ""
            tail = full_text[content_end:span_end_char]
            assert tail in TURN_ENDS, f"row {row_index} turn {span.turn_index}: scored after content {tail!r}"
            # Between the generation prompt and the content: nothing, template whitespace (mistral's " " after
            # [/INST], a BPE token merging the prompt's newline with the content's), or the last-turn decoration.
            assert head.strip() == "" or head in LAST_TURN_DECORATIONS, (
                f"row {row_index} turn {span.turn_index}: scored before content {head!r}"
            )
        # Every assistant content token is scored, so nothing axolotl scored inside content is lost.
        assert content <= new_scored, f"row {row_index}: content tokens unscored"
        lost = old_scored - new_scored
        assert not (lost & content)
        if lost:
            lost_tokens[row_index] = tokenizer.decode([input_ids[k] for k in sorted(lost)])

    if lost_tokens:
        assert fixture.tokenizer_repo in SHIFTED_BY_AXOLOTL, f"tokens axolotl scored and the rule does not: {lost_tokens}"

    # Rows axolotl kept are all still kept, in axolotl's order; the only additions are rows axolotl left all-masked.
    expected = sorted(new_kept, key=lambda item: len(item[1].input_ids))
    assert [row_index for row_index, _ in expected if row_index not in newly_scored_rows] == [
        row.row_index for row in fixture.final
    ], f"newly scored rows: {newly_scored_rows}"
    assert all(old_raw[row_index].dropped == "unscored" for row_index in newly_scored_rows)

    # And the evaluator's build is exactly that list.
    build = build_text_eval_dataset(rows, ChatTemplateDatasetType.model_validate(fixture.dataset_type), tokenizer, config)
    assert [(example.input_ids, example.labels) for example in build.examples] == [
        (result.input_ids, result.labels) for _, result in expected
    ]



# kanana renders an assistant-first conversation, but not that assistant turn on its own ("No user query found"). axolotl
# rendered turn prefixes while loading, so the whole file failed; the rule only needs that render for its turn-end lookup,
# so the row is a per-row RENDER drop and the rest of the eval set is scored.
REJECTED_ROW_BECOMES_RENDER_DROP = {"kakaocorp__kanana-2-3b-instruct__chat", "kakaocorp__kanana-2-3b-instruct__chat_seq256"}


@pytest.mark.parametrize("path", [path for path in CHAT_FIXTURES if _load(path).excluded_rows], ids=lambda path: path.stem)
def test_rows_the_template_rejects(path: Path, tmp_path: Path):
    """axolotl could not load a file holding a row the template raises on, so the eval failed. That stays true whenever the
    template refuses the conversation itself; a row only an auxiliary render rejects becomes a RENDER drop."""
    fixture = _load(path)
    tokenizer = _eval_tokenizer(fixture)
    rows = _eval_rows(fixture, tmp_path, list(range(len(_source_rows(fixture)))))
    dataset_type = ChatTemplateDatasetType.model_validate(fixture.dataset_type)
    if path.stem in REJECTED_ROW_BECOMES_RENDER_DROP:
        build = build_text_eval_dataset(rows, dataset_type, tokenizer, _config(fixture))
        assert build.dropped[DropReason.RENDER] == len(fixture.excluded_rows)
        return
    with pytest.raises(Exception):  # noqa: B017 - the template's own error type (jinja TemplateError, ...)
        build_text_eval_dataset(rows, dataset_type, tokenizer, _config(fixture))


def _scored_text(fixture: GoldenFixture, tokenizer, tmp_path: Path, row_index: int) -> list[str]:
    [row] = _eval_rows(fixture, tmp_path, [row_index])
    result = label_chat(_chat_tokenizer(fixture, tokenizer), _messages(fixture, row))
    assert isinstance(result, ChatLabelled)
    return [tokenizer.decode(result.input_ids[span.start : span.end]) for span in result.spans]


def test_qwen3_base_two_turn_row_is_corrected(tmp_path: Path):
    """Defect 2 fixed on the continuous-SFT lineage stand-in: each turn scores its own content and its turn end."""
    fixture = _load(FIXTURE_DIR / "Qwen__Qwen3-0.6B-Base__chat.json")
    tokenizer = _eval_tokenizer(fixture)
    assert _scored_text(fixture, tokenizer, tmp_path, 1) == [
        "Paris is the capital.<|im_end|>\n",
        "<think>\n\n</think>\n\nBerlin.<|im_end|>\n",
    ]


def test_qwen25_base_now_scores_end_of_turn(tmp_path: Path):
    """Defect 1 fixed: <|im_end|> is scored on every assistant turn although eos is <|endoftext|>."""
    fixture = _load(FIXTURE_DIR / "Qwen__Qwen2.5-0.5B__chat.json")
    tokenizer = _eval_tokenizer(fixture)
    [row] = _eval_rows(fixture, tmp_path, [1])
    result = label_chat(_chat_tokenizer(fixture, tokenizer), _messages(fixture, row))
    assert isinstance(result, ChatLabelled)
    assert [label for label in result.labels if label != IGNORE_INDEX].count(QWEN_IM_END) == 2


def _drop_reason_for(result: ChatLabelled, config) -> DropReason | None:
    example = TokenizedExample(input_ids=result.input_ids, attention_mask=[1] * len(result.input_ids), labels=result.labels)
    return _drop_reason(example, config)
