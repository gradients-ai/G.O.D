# Obfuscation detection

The detector that tournament repository validation runs on every miner's training repository at enrolment. Called from `validator/tournament/github_validation.py`.

## Contents

| Path | Purpose |
| --- | --- |
| `anti_obfuscation` | Bundled detector binary (`OBFUSCATION_DETECTION_PATH`). |
| `__init__.py` | Package marker. |

## See also

- [Training repository](../../../docs/miner.md#training-repository): the readable-source rule this check enforces.
