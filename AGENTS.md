# AGENTS.md

This file applies to the entire repository. Use it as the operating manual for agentic coding in G.O.D.

## What this repo is

G.O.D is the Gradients on Demand subnet runtime: a Python system for Bittensor subnet training jobs and tournaments. It is not a frontend app. Its runtime boundaries are `core/`, `miner/`, `trainer/`, `validator/`, `subnet_api/`, `ops/`, `tests/` and `docs/`. What each holds is in `docs/developer.md`, section "Repository layout".

The product-facing mental model is in `SKILL.md`; update it when public Gradients API behaviour, pricing, account flow, task types, or tournament user expectations change.

## Start by reading

Before editing, read the smallest set that explains the domain you are changing:

- Repo overview and commands: `README.md`, `docs/developer.md`, `docs/README.md`.
- Miner contract: `docs/miner.md` (hub) and the tournament guides `docs/miner_text.md`, `docs/miner_image.md`, `docs/miner_compression.md`, plus `miner/endpoints/training_repo.py` and `core/models/payload_models.py`.
- Shared contracts: `core/README.md`, `core/models/README.md`, `core/constants/README.md`.
- Validator API/tasks: `validator/README.md`, `validator/endpoints/README.md`, `validator/tasks/README.md`.
- DB changes: `validator/db/README.md`, `validator/db/migrations/README.md`, `validator/db/constants.py`, matching `validator/db/sql/*.py`.
- Evaluation/scoring/tournaments: `validator/evaluation/README.md`, `validator/scoring/README.md`, `validator/tournament/README.md`, `validator/lifecycle/README.md`.
- Trainer runtime: `trainer/README.md`, `trainer/containers/README.md`, `trainer/model_prep/README.md`, `trainer/quant_decode/README.md`.
- Ops/config/deploy tooling: `ops/README.md`, `ops/tools/README.md`, relevant subdirectory README.
- Tests: `tests/README.md`, plus the relevant runtime README under `tests/` where one exists.

## Design rules

- Put code in the domain that owns it. Do not create new catch-all modules such as `utils.py`, `shared.py`, or generic helper packages unless the surrounding package already owns that exact concern.
- Keep `core/` only for contracts or helpers used by more than one runtime. Validator-only schemas belong in `validator/`; trainer-only behaviour belongs in `trainer/`; miner behaviour belongs in `miner/`.
- Prefer the narrowest existing module over a new abstraction. Add abstraction only when it removes real duplication or clarifies a cross-module contract.
- Keep constants near their owner. Use `core/constants/*` only for shared constants, `validator/*/constants.py` for validator-domain constants, and `trainer/constants.py` for trainer runtime constants.
- Preserve public contracts deliberately. If a Pydantic model, endpoint path, task type, enum value, CLI argument, Docker path, env var, or DB column changes, update every caller, docs, tests, and operational references in the same change.
- Use structured models and parsers instead of ad hoc string manipulation. Pydantic models should carry validation and examples for request/response data where useful.
- Keep async boundaries clean. Existing FastAPI, asyncpg, Redis, httpx, and Docker orchestration patterns should be followed instead of introducing incompatible blocking flows.
- Never log secrets, GitHub tokens, wallet secrets, API keys, Hugging Face tokens, signed URLs, or private dataset contents. Treat env files and task payloads as potentially sensitive.
- Avoid unrelated refactors. This repo coordinates live services, tournaments, payments, training containers, and scoring; narrow changes are easier to verify and safer to deploy.

## Change completeness checklist

Use this checklist whenever the change touches more than a private helper:

- Public API or task request changes: update `core/models/payload_models.py`, endpoint code under `validator/endpoints/`, client/service constants if relevant, `SKILL.md`, `docs/developer.md`, and tests.
- Miner tournament contract changes: update `core/models/payload_models.py`, `core/models/tournament_models.py`, `miner/endpoints/training_repo.py`, `docs/miner.md` or the affected tournament guide, and any tournament validation tests.
- Trainer container contract changes: update `trainer/runtime.py`, `trainer/training_paths.py`, trainer entrypoints, Docker/ops docs, `docs/miner.md` or the affected tournament guide, and trainer tests.
- DB schema changes: add a new timestamped migration in `validator/db/migrations/`; do not edit applied migrations. Update `validator/db/constants.py`, SQL access modules, task/model schemas, tests, and docs if operator-visible.
- Tournament/scoring changes: update constants, models, state-machine code, reports/analytics endpoints if affected, simulations or probes when useful, and focused tests under `tests/validator/tournament` or `tests/validator/scoring`.
- Evaluation changes: update evaluation runtime/model/result-processing code together, including text, image and compression assumptions, and add focused tests under `tests/validator/evaluation`.
- Config/env changes: update `ops/tools/config/create_config.py` and related config models/validation, Taskfile commands if relevant, `.env` documentation in `docs/developer.md`, and ops README/runbook references.
- Operational tool changes: update the relevant `ops/tools/*/README.md`, keep command names discoverable, and avoid requiring hidden local state.
- New user-visible behaviour: update the nearest README plus `docs/developer.md` or the miner guides when the behaviour is developer/miner-facing.

## Python and style

- Python version is `>=3.10`; dependencies are declared in `pyproject.toml`.
- Ruff is the source of style truth: line length is 130, lint selects `E`, `F`, and `I`, and imports are force-single-line with two blank lines after imports.
- Prefer `core.logging.get_logger` or the existing module logger pattern.
- SQL modules should use `validator.db.constants` for table/column names and parameterised query values for user data.
- Keep Pydantic validation close to the model that owns the data. Convert client mistakes into clear `HTTPException(status_code=400, detail=...)` at API boundaries.
- Do not add broad dependencies without checking whether an existing dependency or standard library tool already fits.

## Verification

Run the smallest reliable checks for the touched area, then broaden when the blast radius is large.

The lint and test commands are in `docs/developer.md`, section "Testing and linting". Run focused tests with `-o addopts=''` to turn off random ordering.

For pure reorganisation, also consider import smoke or compile checks. If you cannot run an important test because Docker, GPU, network credentials, or external services are unavailable, say that explicitly in the final note and name the command you would run.

## Documentation standard

- Keep docs updated in the same change as code. Do not leave behaviour discoverable only from code.
- Prefer updating existing maintained docs over adding new top-level docs. The maintained set in `docs/` is the index `README.md` plus `developer.md`, `miner.md`, `miner_text.md`, `miner_image.md`, `miner_compression.md` and `organic-package.md`.
- Each fact has one home. Other files link to it and do not restate it.
- The README stays a brief front door. Depth goes in `docs/` and the module READMEs.
- Miner docs are a hub (`docs/miner.md`) plus one guide per tournament. The three guides share the same nine `##` sections in the same order.
- Naming: text, image and compression are tournaments, always listed in that order. "Track" means the fixed or freeform kind of compression task. `quant` appears only as the API value and in code identifiers and paths.
- Describe the system as it is. No wording that explains a rule through an earlier design.
- Style: sentence-case headings, British spelling, `1%` and "rounds 1-3", module READMEs on the `## Contents` Path/Purpose template (Path/Covers is also allowed in tests READMEs).
- Module READMEs should stay short and structural: what belongs there, what the files do, and where to look next.
- When adding new operational commands or tools, document the purpose, required env vars, example invocation, and expected side effects.
- When changing public API payloads, include field meaning, defaults, and compatibility notes in docs or model descriptions.

## Agent workflow

1. Inspect before changing. Use `rg`/`rg --files` and read nearby README files, models, tests, and call sites.
2. Make the smallest coherent change that keeps contracts, docs, and tests aligned.
3. Preserve user changes and unrelated work in the tree. Do not revert files you did not intentionally edit.
4. Add or update focused tests for changed behaviour, especially scoring, tournament advancement, DB persistence, API validation, dataset mapping, and trainer runtime paths.
5. Run targeted verification. Run full lint/tests when touching shared contracts, shared constants, task models, scoring, migrations, or cross-runtime behaviour.
6. In the final response, summarise the change, mention tests run, and call out any skipped checks or operational follow-up.
