# Bundled Chat Templates

Chat templates a text task can select by name instead of the tokenizer's own (`ChatTemplateDatasetType.chat_template`),
resolved by `validator/evaluation/chat_templates.py`.

## Contents

- `chatml.jinja`: the `chatml` template (default for API-created chat tasks). Byte-identical to the one the axolotl-based
  eval bundled (axolotl 0bda5a13); `tests/validator/evaluation/test_chat_templates.py` pins its sha256. Do not edit.
