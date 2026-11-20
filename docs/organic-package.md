# Chain-free organic package

How to install this repository as a library for organic (paid) training jobs without the chain dependencies. For developers who embed the orchestration code in another service.

## Installation

Pin a reviewed commit SHA or a protected release tag. Do not install from a branch.

Contracts and the dstack client:

```bash
pip install "Gradients-on-Demand @ git+https://github.com/gradients-ai/G.O.D@<reviewed-commit>"
```

Dataset preparation and task SQL:

```bash
pip install "Gradients-on-Demand[organic] @ git+https://github.com/gradients-ai/G.O.D@<reviewed-commit>"
```

A validator installs the extras it runs (`task install`):

```bash
pip install ".[organic,chain,evaluation-remote,validator]"
```

## Extras

| Extra | Covers |
| --- | --- |
| none | API contracts, HTTP client, logging. |
| `organic` | Dataset preparation, Postgres, Redis, object storage, Hugging Face access. |
| `evaluation-remote` | The remote evaluation SDK. Install only in a worker image that runs remote evaluation. |
| `chain` | Wallet and chain access. |
| `validator` | The validator API and its services. |
| `subnet-api` | The public API in [subnet_api/](../subnet_api/README.md). |
| `gpu` | Local evaluation dependencies. |
| `dev` | Tests, lint and pre-commit. |

A chain-free worker does not install `chain` or `gpu`.

## Entry points

| Module | Purpose |
| --- | --- |
| `core.models.payload_models` | API contracts. |
| `validator.infrastructure.dstack_client` | dstack HTTP client. |
| `validator.infrastructure.organic_dstack` | Organic run plans. The training image must be pinned by digest. |
| `validator.tasks.requests` | Text, image and compression task preparation. |
| `validator.db.sql.tasks` | Task SQL. |
| `validator.scoring.organic` | `evaluate_organic_repo`: evaluates one trained repository without a database. Needs `evaluation-remote` when called. |

Importing these modules does not need the `chain` extra and does not open a wallet or chain connection. `tests/test_organic_package.py` checks this for every module above except `validator.db.sql.tasks`.
