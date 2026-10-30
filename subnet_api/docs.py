"""Public API reference: OpenAPI metadata and the interactive /docs page."""

import json
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse


SCALAR_VERSION = "1.76.0"
SCALAR_INTEGRITY = "sha384-oH1C0RwqcwbVXkC+r8A+4qRoyEUQdtxAzOwumkxZtQBBcsy61FOoPt7B50OVJYYQ"

DESCRIPTION = """
Public, read-only data for Bittensor subnet 56, Gradients: every task the validator scored,
every tournament and its results, and how emissions are split between miners.

Everything here is what the validator itself used, so you can audit its weights instead of
trusting them.

## Access

No account or key is needed. Every endpoint is a plain `GET`:

```bash
curl https://api.sn56.io/tournament/latest/details
```

* Responses are cached for **30 seconds**, so polling faster returns the same data.
* Send `Accept-Encoding: gzip`; tournament details are large and compress well.
* List endpoints page with `limit` (1–100, default 100) and `page` (from 1).

## Quickstart

```bash
# The tournaments currently deciding emissions
curl https://api.sn56.io/tournament/latest/details

# One tournament, round by round
curl https://api.sn56.io/tournament/<tournament_id>/details

# Recent tasks a miner was scored on
curl "https://api.sn56.io/auditing/tasks/hotkey/<hotkey>?limit=20"

# Who is earning what right now
curl https://api.sn56.io/v1/performance/latest-tournament-weights
```

## How emissions work

Miners compete in **tournaments**, one per track: text, image and environment. Each
tournament runs in rounds of training tasks; the winner becomes the track's **champion**
and earns that track's share of emissions, which decays over time until someone beats
them by enough to take the title. The *Emissions* endpoints show the current split and
project what a challenger would earn.

## Errors

Errors are JSON: `{"detail": "<what went wrong>"}`.

| Status | Meaning |
|---|---|
| `404` | No such task, tournament or coldkey |
| `422` | A path or query parameter is malformed, e.g. not an SS58 address |
| `502` | The validator is unreachable or returned something unexpected; retry shortly |
"""

TAGS: list[dict[str, str]] = [
    {
        "name": "Auditing",
        "description": (
            "Every task the validator ran and how each miner scored on it, plus the scores file "
            "behind the weights it last set on chain."
        ),
    },
    {
        "name": "Tournaments",
        "description": "Tournament results, round by round, and the TAO fees for entering one.",
    },
    {
        "name": "Emissions",
        "description": "How emissions are split between champions now, and what a challenger would earn.",
    },
]

TAG_GROUPS = [{"name": "Subnet 56", "tags": ["Tournaments", "Emissions", "Auditing"]}]

_PAGED = " Paged with `limit` and `page`, newest first."

OPERATIONS: dict[tuple[str, str], dict[str, Any]] = {
    ("get", "/tournament/latest/details"): {
        "tags": ["Tournaments"],
        "summary": "Latest tournaments",
        "description": (
            "The most recent tournament of each track, with participants, rounds, final scores and the weight each track carries."
        ),
    },
    ("get", "/tournament/{tournament_id}/details"): {
        "tags": ["Tournaments"],
        "summary": "Get a tournament",
        "description": ("One tournament: who entered, every round and its tasks, head-to-head results and the final ranking."),
    },
    ("get", "/tournament/fees"): {
        "tags": ["Tournaments"],
        "summary": "Entry fees",
        "description": "The fee to enter each tournament track, in RAO (1 TAO = 10⁹ RAO).",
    },
    ("get", "/tournament/balance/{coldkey}"): {
        "tags": ["Tournaments"],
        "summary": "Coldkey fee balance",
        "description": (
            "TAO a coldkey has sent the validator to pay tournament fees, and what is left of it, "
            "in RAO. Returns `404` for a coldkey that never sent any."
        ),
    },
    ("get", "/v1/performance/latest-tournament-weights"): {
        "tags": ["Emissions"],
        "summary": "Current weights",
        "description": (
            "The emission split the validator is setting now: each track's weight, how much is "
            "burned, and the top miners of each track with their share."
        ),
    },
    ("get", "/v1/performance/weight-projection"): {
        "tags": ["Emissions"],
        "summary": "Project a challenger's earnings",
        "description": (
            "If a challenger beat the current champion by `percentage_improvement` percent, "
            "the weight and alpha they would earn over the following days, per track."
        ),
    },
    ("get", "/v1/performance/weight-projection-static"): {
        "tags": ["Emissions"],
        "summary": "Projections at 5, 10, 15 and 20%",
        "description": "*Project a challenger's earnings* for the four most common improvement levels at once.",
    },
    ("get", "/v1/performance/last-boss-battle"): {
        "tags": ["Emissions"],
        "summary": "Last boss battle",
        "description": (
            "How the latest challenger scored against the defending champion on each task of the final round, per track."
        ),
    },
    ("get", "/auditing/tasks"): {
        "tags": ["Auditing"],
        "summary": "List tasks",
        "description": "Tasks the validator ran, organic and synthetic." + _PAGED,
    },
    ("get", "/auditing/tasks/{task_id}"): {
        "tags": ["Auditing"],
        "summary": "Get a task",
        "description": "One task with every miner that trained on it and their scores.",
    },
    ("get", "/auditing/tasks/hotkey/{hotkey}"): {
        "tags": ["Auditing"],
        "summary": "List a miner's tasks",
        "description": "Tasks a miner hotkey trained on, with their scores." + _PAGED,
    },
    ("get", "/auditing/scores-url"): {
        "tags": ["Auditing"],
        "summary": "Scores behind the last weights",
        "description": (
            "A link to the scores file the validator used the last time it set weights on chain, "
            "so you can recompute them yourself."
        ),
    },
}

PARAMETERS: dict[str, str] = {
    "limit": "Results per page, 1–100.",
    "page": "Page number, from 1.",
    "hotkey": "Miner hotkey, SS58 address.",
    "coldkey": "Coldkey, SS58 address.",
    "task_id": "Task id (UUID).",
    "tournament_id": "Tournament id, e.g. `tourn_7057f4bd8e7bd216_20261026`.",
    "percentage_improvement": "How much better than the champion the challenger scored, in percent.",
}


def build_openapi(app: FastAPI, public_url: str) -> dict[str, Any]:
    schema = get_openapi(
        title="Gradients SN56 API",
        version=app.version,
        summary="Open data for Bittensor subnet 56.",
        description=DESCRIPTION,
        routes=app.routes,
        tags=TAGS,
        servers=[{"url": public_url}],
    )
    for path, item in schema["paths"].items():
        for method, operation in item.items():
            operation.update(OPERATIONS.get((method, path), {}))
            for parameter in operation.get("parameters", []):
                if parameter["name"] in PARAMETERS:
                    parameter["description"] = PARAMETERS[parameter["name"]]
    order = [path for _, path in OPERATIONS]
    schema["paths"] = dict(sorted(schema["paths"].items(), key=lambda item: order.index(item[0])))
    schema["x-tagGroups"] = TAG_GROUPS
    return schema


def docs_page() -> str:
    config = {
        "url": "/openapi.json",
        "layout": "modern",
        "theme": "none",
        "forceDarkModeState": "dark",
        "hideDarkModeToggle": True,
        "hideClientButton": True,
        "hideModels": True,
        "showDeveloperTools": "never",
        "agent": {"disabled": True},
        "mcp": {"disabled": True},
        "telemetry": False,
        "defaultHttpClient": {"targetKey": "shell", "clientKey": "curl"},
        "favicon": "https://gradients.io/favicon.png",
        "customCss": _CSS,
    }
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gradients SN56 API</title>
  <link rel="icon" href="https://gradients.io/favicon.png">
</head>
<body>
  <div id="app"></div>
  <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference@{SCALAR_VERSION}/dist/browser/standalone.js"
          integrity="{SCALAR_INTEGRITY}" crossorigin="anonymous"></script>
  <script>Scalar.createApiReference("#app", {json.dumps(config)})</script>
</body>
</html>"""


_CSS = """
.dark-mode {
  --scalar-background-1: #0a0a0a;
  --scalar-background-2: #121212;
  --scalar-background-3: #1c1c1c;
  --scalar-background-accent: #ffffff12;
  --scalar-color-1: #fafafa;
  --scalar-color-2: #a3a3a3;
  --scalar-color-3: #6b6b6b;
  --scalar-color-accent: #fafafa;
  --scalar-border-color: #232323;
  --scalar-button-1: #fafafa;
  --scalar-button-1-color: #0a0a0a;
  --scalar-button-1-hover: #d4d4d4;
  --scalar-color-green: #4ade80;
  --scalar-color-red: #f87171;
  --scalar-color-yellow: #facc15;
  --scalar-color-blue: #60a5fa;
  --scalar-color-orange: #fb923c;
  --scalar-color-purple: #c084fc;
}
.dark-mode .sidebar { --scalar-sidebar-background-1: #0a0a0a; }
"""


def install_docs(app: FastAPI, public_url: str) -> None:
    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            app.openapi_schema = build_openapi(app, public_url)
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]

    @app.get("/docs", include_in_schema=False)
    async def docs() -> HTMLResponse:
        return HTMLResponse(docs_page())
