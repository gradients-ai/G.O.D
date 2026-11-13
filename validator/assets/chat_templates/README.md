# Bundled chat templates

Chat templates a text task can select by name in place of the tokenizer's template (`ChatTemplateDatasetType.chat_template`). `validator/evaluation/chat_templates.py` resolves the name.

## Contents

| Path | Purpose |
| --- | --- |
| `chatml.jinja` | The `chatml` template. |

> [!IMPORTANT]
> Do not edit `chatml.jinja`. `tests/validator/evaluation/test_chat_templates.py` pins its sha256, because evaluation labels depend on its exact bytes.
