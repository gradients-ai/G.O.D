# Auditing

Auditor scripts that check a validator's weights from outside the validator service.

## Contents

| Path | Purpose |
| --- | --- |
| `audit.py` | Auditor runner: recomputes weights from published task results and compares them with the validator's. |
| `audit_no_hotkey.py` | The same audit, run from a minimal config without a registered hotkey. |
