# Steps to take to update the validator automatically
# Change each time but take caution


. $HOME/.venv/bin/activate

# Package install (extras + cyscale handling) lives in `task install`.
# Only restart the validator if the install succeeded so a broken venv does not
# replace a still-running process.
task install && task validator

# Update observability server - redeploy if configs changed
if git diff HEAD~1 HEAD --name-only 2>/dev/null | grep -qE "grafana-training|loki-training|observability-server|vector/vector.toml"; then
    echo "Observability configs changed, redeploying observability server..."
    task deploy-observability-server
fi
