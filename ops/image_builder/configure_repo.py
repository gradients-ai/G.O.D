"""Configure the snapshot repo in Woodpecker and commit the build recipe to it (`setup.sh configure-repo`).

Run once after the first Woodpecker login, and again whenever woodpecker.yml changes. Reads .env (via setup.sh):
BUILD_BOX_HOST, WOODPECKER_TOKEN, SNAPSHOT_REPO, SNAPSHOT_TOKEN. Prints the IMAGE_BUILD_WOODPECKER_REPO_ID the validator
needs. Standard library only, so it runs on a bare host.
"""

import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent


def call(url: str, token: str, method: str = "GET", body: dict | None = None) -> dict:
    request = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {url.split('?')[0]} failed: {e.code} {e.read()[:300]!r}") from None
    return json.loads(raw) if raw else {}


def main() -> None:
    env = {k: os.environ.get(k, "") for k in ("BUILD_BOX_HOST", "WOODPECKER_TOKEN", "SNAPSHOT_REPO", "SNAPSHOT_TOKEN")}
    missing = [k for k, v in env.items() if not v]
    if missing:
        raise SystemExit(f"missing in .env: {', '.join(missing)}")
    match = re.match(r"https://github\.com/([^/]+)/([^/]+?)(\.git)?/?$", env["SNAPSHOT_REPO"])
    if not match:
        raise SystemExit("SNAPSHOT_REPO must be https://github.com/<owner>/<repo>.git")
    owner, name = match.group(1), match.group(2)
    github = f"https://api.github.com/repos/{owner}/{name}"
    woodpecker = f"https://{env['BUILD_BOX_HOST']}/api"
    gh_token, wp_token = env["SNAPSHOT_TOKEN"], env["WOODPECKER_TOKEN"]

    # The recipe on main, which must be the default branch (Woodpecker reads it from there).
    recipe = (HERE / "woodpecker.yml").read_text()
    try:
        current = call(f"{github}/contents/.woodpecker.yml?ref=main", gh_token)
    except SystemExit:
        current = {}
    if base64.b64decode(current.get("content", "")).decode() != recipe:
        body = {"message": "Update the tournament image build recipe", "content": base64.b64encode(recipe.encode()).decode()}
        if current.get("sha"):
            body["sha"] = current["sha"]
        else:
            body["branch"] = "main"
        call(f"{github}/contents/.woodpecker.yml", gh_token, "PUT", body)
        print("recipe committed to main")
    call(github, gh_token, "PATCH", {"default_branch": "main"})

    forge_id = call(github, gh_token)["id"]
    try:
        repo = call(f"{woodpecker}/repos/lookup/{owner}/{name}", wp_token)
    except SystemExit:
        repo = call(f"{woodpecker}/repos?forge_remote_id={forge_id}", wp_token, "POST")
    repo_id = repo["id"]
    # Trusted for volumes: the recipe mounts the Docker socket and the builder config. Only the API starts builds.
    call(
        f"{woodpecker}/repos/{repo_id}",
        wp_token,
        "PATCH",
        {
            "timeout": 60,
            "allow_pr": False,
            "allow_deploy": False,
            "trusted": {"volumes": True, "network": False, "security": False},
            "cancel_previous_pipeline_events": [],
        },
    )
    secret = {"name": "snapshot_read_token", "value": gh_token, "events": ["manual"], "images": []}
    existing = {s["name"] for s in call(f"{woodpecker}/repos/{repo_id}/secrets", wp_token) or []}
    if "snapshot_read_token" in existing:
        call(f"{woodpecker}/repos/{repo_id}/secrets/snapshot_read_token", wp_token, "PATCH", secret)
    else:
        call(f"{woodpecker}/repos/{repo_id}/secrets", wp_token, "POST", secret)
    print(f"Woodpecker repo configured. Validator setting: IMAGE_BUILD_WOODPECKER_REPO_ID={repo_id}")


if __name__ == "__main__":
    sys.exit(main())
