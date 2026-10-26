# Chain-free organic package

The `organic-package` branch separates reusable organic orchestration from validator-only dependencies.

## Installation

Contracts and the dstack client:

```bash
pip install "Gradients-on-Demand @ git+https://github.com/gradients-ai/G.O.D@<reviewed-commit>"
```

Dataset preparation and organic SQL dependencies:

```bash
pip install "Gradients-on-Demand[organic] @ git+https://github.com/gradients-ai/G.O.D@<reviewed-commit>"
```

Add `[evaluation-remote]` only to a worker image that supports remote evaluation. The Gradients product worker must not install the `chain` or `gpu` extras.

The validator installs the extras it actually runs:

```bash
pip install ".[organic,chain,evaluation-remote,validator]"
```

Production consumers pin a reviewed commit SHA or a protected release tag. They must not follow the mutable `organic-package` branch.

## Public organic entry points

- `core.models.payload_models`: API contracts.
- `validator.infrastructure.dstack_client`: dstack HTTP client.
- `validator.infrastructure.organic_dstack`: explicit, digest-pinned organic run plans.
- `validator.tasks.requests`: text and image preparation.
- `validator.db.sql.tasks`: task SQL; chain node models are imported only by chain-specific functions.
- `validator.scoring.organic`: database-free evaluation of one trained repository (`evaluate_organic_repo`). Requires the `evaluation-remote` extra at call time.

Importing these modules does not initialize a wallet or substrate connection. Chain-specific operations still require the `chain` extra and fail only when invoked without it.
