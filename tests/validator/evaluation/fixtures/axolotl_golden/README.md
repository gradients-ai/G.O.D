# Axolotl golden fixtures (0bda5a13)

What the axolotl-based text eval produced, frozen once on real tokenizers so the axolotl-free implementation can be proven
against it: bit-identical for instruct and completion, different only by the intended deltas for chat. Captured by
`../../golden/capture_axolotl_golden.py` (see `../../golden/README.md`) under axolotl `0bda5a13`, transformers `5.12.1`,
tokenizers `0.22.2`, datasets `4.8.5`, torch `2.9.1+cpu`. Do not edit by hand; `../../test_axolotl_golden_fixtures.py`
checks every file is internally consistent.

## Contents

- `datasets/`: the hand-written source rows (standard eval JSON columns only, since axolotl maps every column).
  Cases are listed by row index below.
- `<tokenizer_slug>__<kind>.json`: one fixture per tokenizer and kind (`slug` = repo id with `/` replaced by `__`).

## Capture command

```
cd <G.O.D checkout at the capture commit>
HF_HUB_OFFLINE=1 PYTHONPATH=. <golden venv>/bin/python -m tests.validator.evaluation.golden.capture_axolotl_golden
```

## Fixture format

`GoldenFixture` (declared in the capture script, re-declared strictly in the test):

- provenance: `axolotl_commit`, `transformers_version`, `datasets_version`, `tokenizer_repo`, `tokenizer_sha` (HF snapshot
  commit), `tokenizer_files_sha256`, `bos/eos/pad_token_id` (pad after the evaluator's pad := eos fallback).
- case: `kind`, `dataset_type_name` + `dataset_type` (the `InstructTextDatasetType` / `ChatTemplateDatasetType` given to
  `EvaluationArgs`), `max_position_embeddings` (fake model config fed to the halving rule), resulting `sequence_len`,
  `rows_file` (relative to this directory).
- `excluded_rows`: rows whose rendering raises inside axolotl. A raise fails the whole axolotl load, so those rows were
  recorded with the exception and the rest were captured from a file without them (original `row_index` kept).
- `raw`: axolotl's tokenised output per source row before any filtering, in source order, `{row_index, chunk, input_ids,
  labels, dropped}`. `chunk` > 0 only for completion rows split into `sequence_len` pieces. `dropped` is `null` for kept rows,
  else `empty` (the strategy emitted no row: empty/None completion text), `short` (< 2 tokens), `long` (> `sequence_len`),
  or `unscored` (every label -100; G.O.D's filter).
- `final`: the exact output of `instruct_text._load_evaluation_dataset`, i.e. kept `raw` rows stably sorted by length.
  The capture refuses to write a fixture unless this equality holds. `attention_mask` is all ones everywhere (asserted at
  capture) and not stored.

## Kinds

| kind | rows file | dataset type | sequence_len |
|---|---|---|---|
| `instruct` | `instruct_rows.json` | `field_instruction=instruct, field_input=input, field_output=output` | 4096 |
| `instruct_seq256` | `instruct_rows.json` | same | 256 |
| `instruct_system` | `instruct_system_rows.json` | + `field_system=system` | 4096 |
| `instruct_system_nofield` | `instruct_system_rows.json` | no `field_system` (axolotl's default `"system"` still reads the column) | 4096 |
| `instruct_system_prompt` | `instruct_system_rows.json` | + `field_system=system, system_prompt="Default system prompt.", system_format="[SYS] {system} [/SYS] "` | 4096 |
| `instruct_custom_format` | `instruct_rows.json` | + `format="Q: {instruction}\nI: {input}\nA:", no_input_format="Q: {instruction}\nA:"` | 4096 |
| `completion` | `completion_rows.json` | `field_instruction=instruct` only (completion strategy) | 4096 |
| `completion_seq256` | `completion_rows.json` | same | 256 |
| `chat` | `chat_rows.json` | `chat_template=tokenizer_default`, `conversations` of `{from, value}`, roles `user`/`assistant` | 4096 |
| `chat_seq256` | `chat_rows.json` | same | 256 |
| `chat_sharegpt` | `chat_sharegpt_rows.json` | same, `chat_user_reference=human`, `chat_assistant_reference=gpt` | 4096 |
| `chat_chatml` | `chat_rows.json` | `chat_template=chatml` (axolotl's bundled template) | 4096 |

Chat kinds are captured only for tokenizers with a chat template; `chat_chatml` only on Qwen2.5-0.5B and
Llama-3.2-1B-Instruct.

## Dataset cases

`instruct_rows.json`: 0 input present; 1 input `""`; 2 input null; 3 output `""`; 4 output null; 5 output key missing
(pyarrow fills null); 6 instruction `""`; 7-12 output ending in an eos text (`<|endoftext|>`, `<|im_end|>`, `</s>`,
`<|eot_id|>`, `<eos>`, `<|end_of_text|>`); 13 prompt > 256 tokens; 14 response > 256 tokens; 15 unicode/emoji; 16 leading/
trailing whitespace; 17 `{` braces in every field; 18 multi-line fields; 19 whitespace-only input; 20 everything empty;
21 empty prompt, one-character output; 22 code block; 23 plain short row.

`instruct_system_rows.json`: 0 system text; 1 system `""`; 2 system null; 3 system text + input; 4 system null + input;
5 system with surrounding spaces; 6 system key missing (pyarrow fills null); 7 braces in system and output; 8 output `""`.

`completion_rows.json`: 0 short text; 1 text > 256 tokens (chunked at 256); 2 `""`; 3 one character; 4 trailing newline;
5 ends with eos text; 6 unicode; 7 null.

`chat_rows.json`: 0 single turn; 1 two assistant turns (France/Germany, the defect probe); 2 three assistant turns;
3 system + one turn; 4 system + two turns; 5 assistant content with leading `\n`; 6 trailing `\n`; 7 `\n\n` both sides;
8 empty assistant content; 9 empty first assistant turn, then a real one; 10 user-only; 11 assistant > 256 tokens;
12 user > 256 tokens; 13 unicode/emoji; 14 jinja-looking text in content; 15 conversation ending on a user turn;
16 assistant speaks first; 17 whitespace-only assistant content; 18 code block.

`chat_sharegpt_rows.json`: 0 single turn; 1 two assistant turns; 2 system + one turn; 3 user-only; 4 empty assistant content.

## Recorded facts

- Excluded rows (the template raises, so axolotl could not load a file containing them; the eval would fail the task):
  gemma-2-2b-it `chat`/`chat_seq256` rows 3, 4 (`System role not supported`) and 16 (`Conversation roles must alternate`),
  `chat_sharegpt` row 2 (system); mistral-7b-instruct-v0.3 `chat`/`chat_seq256` row 16 (alternation); kanana-2-3b-instruct
  `chat`/`chat_seq256` row 16 (`No user query found in messages`).
- Defect 1 (end of turn never scored when `eos_token` is not the template's turn end): Qwen2.5-0.5B `chat` has no
  `<|im_end|>` (151645) in any label; Qwen2.5-0.5B-Instruct (eos `<|im_end|>`) scores it once per assistant turn.
- Defect 2 (Qwen3 earlier turns shifted by the last-turn-only empty `<think>` block): Qwen3-0.6B-Base `chat` row 1 scores
  `. <|im_end|> \n <|im_start|> user Berlin .` (`[13, 151645, 198, 151644, 872, 94409, 13]`); rows 2, 4, 15 show the same
  shift (`<|im_start|> user` scored instead of content).
- A null `system` value does not fall back to `system_prompt`: axolotl uses `system_prompt` only when the system column
  is absent from the dataset, and pyarrow gives every row the column (null for missing keys), so null renders like `""`
  (`instruct_system_prompt` rows 1, 2, 6).

## Tokenizers

| repo | snapshot sha | bos | eos | pad |
|---|---|---|---|---|
| `Qwen/Qwen3-0.6B-Base` (chat template byte-identical to Qwen3-8B-Base) | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | - | 151643 | 151643 |
| `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | - | 151643 | 151643 |
| `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | - | 151645 | 151643 |
| `unsloth/Llama-3.2-1B-Instruct` | `5a8abab4a5d6f164389b1079fb721cfab8d7126c` | 128000 | 128009 | 128004 |
| `unsloth/Llama-3.2-1B` (instruct kinds only) | `9535bd9b1d1dea6acafbdc4813b728796aeb28da` | 128000 | 128001 | 128004 |
| `unsloth/mistral-7b-instruct-v0.3` | `60b99dcbc7f853b7cd18472a607e8d045b96f03e` | 1 | 2 | 770 |
| `unsloth/gemma-2-2b-it` | `457f2e15bf550c227ce6ad86e2ec108d3e42c106` | 2 | 1 | 0 |
| `unsloth/gemma-2-2b` (instruct kinds only) | `25319945f7fd83b8b903e12081777b7eef2ba993` | 2 | 1 | 0 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | 1 | 2 | 2 |
| `ibm-granite/granite-4.1-3b` | `c0650403e44e78ec0262dab1c90914c65b196c4e` | 100257 | 100257 | 100256 |
| `LiquidAI/LFM2.5-2.6B` | `654f9463ce32b05d0429d76fe1f580b27d4c1ac0` | 124894 | 124900 | 124893 |
| `kakaocorp/kanana-2-3b-instruct` | `6a5d7889964c4c590299d16e309eabab1f73f8a9` | 128000 | 128010 | 128001 |
| `Nanbeige/Nanbeige4.1-3B` | `9c4555c37921c0982af8fffeee1e0a6e6b953c4f` | 166100 | 166101 | 0 |
| `tiiuae/Falcon-H1-0.5B-Base` (instruct kinds only) | `59fb76e8c5d3fc7441b062be638e1ba0afd5c687` | - | 11 | 11 |
| `microsoft/phi-2` (instruct kinds only) | `810d367871c1d460086d9f82db8696f2e0a0fcd0` | 50256 | 50256 | 50256 |
