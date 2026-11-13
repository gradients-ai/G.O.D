# Validator app

Application wiring for the validator service.

## Contents

| Path | Purpose |
| --- | --- |
| `config.py` | Loads the validator config from the environment into the shared `Config` object. |
| `dependencies.py` | FastAPI dependencies: the config and the API key check. |
