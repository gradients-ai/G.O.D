# Axolotl golden capture

One-off tooling that froze what the axolotl-based text eval produced, so the axolotl-free replacement can be proven
bit-identical (instruct, completion) or different only by design (chat). The captured fixtures and their datasets live in
`../fixtures/axolotl_golden/`; the integrity tests are `../test_axolotl_golden_fixtures.py`.

## Contents

- `capture_axolotl_golden.py`: runs G.O.D's real `_load_and_update_evaluation_config` + `instruct_text._load_evaluation_dataset`
  on axolotl 0bda5a13 for every (tokenizer, case) and writes one fixture per pair. Also records axolotl's unfiltered per-row
  output so every dropped row carries its reason, and refuses to write a fixture whose `final` is not exactly the drop rules
  applied to `raw`. Never imported by pytest (it imports axolotl at module level).
- `__init__.py`: package marker, so the script runs with `python -m`.

## Running it

Not part of the test suite and not meant to be re-run casually: the fixtures are the record of the axolotl behaviour the
eval used to have, and recapturing them against a different axolotl or transformers would silently move the reference.
Recapture only to add a tokenizer or a case, in the golden venv (torch 2.9.1+cpu, transformers 5.12.1, tokenizers 0.22.2,
datasets 4.8.5, accelerate 1.13.0, peft 0.19.1, trl 1.5.1, axolotl installed `--no-deps -e` from a checkout of commit
`0bda5a13`; the script checks the axolotl commit and transformers version and refuses to run otherwise):

```
cd <G.O.D checkout>
HF_HUB_OFFLINE=1 PYTHONPATH=. <golden venv>/bin/python -m tests.validator.evaluation.golden.capture_axolotl_golden \
    [--only <tokenizer repo> ...] [--kinds <kind> ...]
```

Tokenizers are read from the local Hugging Face cache (`HF_HUB_OFFLINE=1` is required); download only tokenizer files, e.g.
`huggingface_hub.snapshot_download(repo, allow_patterns=["tokenizer*", "special_tokens_map.json", "chat_template*", "*.model",
"vocab.json", "merges.txt", "added_tokens.json", "config.json"])`. Each fixture records the snapshot sha and the sha256 of
every tokenizer file it was captured with.
