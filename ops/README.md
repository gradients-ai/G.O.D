# Ops

Operational files for validator and trainer operators: image builds, Compose stacks, host scripts, observability config and command-line tools.

## Contents

| Path | Purpose |
| --- | --- |
| `auditing/` | Auditor scripts. |
| `compose/` | Docker Compose stacks for the validator, local development, observability and trainer log shipping. |
| `devcontainer/` | Development container configuration. |
| `docker/` | Dockerfiles, Docker-only requirements, patches and build scripts. |
| `editor/` | Editor configuration. |
| `examples/` | Example HTTP requests and local training runs. |
| `maintenance/` | Host-level maintenance scripts. |
| `observability/` | Grafana, Loki, Prometheus, Tempo, Vector and nginx configuration. |
| `runbooks/` | Debugging guides for operators. |
| `tools/` | Python and shell tools grouped by domain. |
| `trainer_ops/` | Trainer host service installation. |
| `validator_ops/` | Validator startup, auto-update, manual evaluation and Grafana setup. |

## See also

- [Developer guide](../docs/developer.md): setup, configuration and running the services.
