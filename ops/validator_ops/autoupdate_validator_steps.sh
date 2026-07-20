# Steps to take to update the validator automatically
# Change each time but take caution


. $HOME/.venv/bin/activate

# fiber v2.7.0+ moved to async-substrate-interface 2.x, which depends on `cyscale`
# instead of `scalecodec`/`substrate-interface` (same import namespace as scalecodec,
# so it refuses to load if the old packages are still around). `pip install -e .`
# won't remove those on its own since they're no longer required by anything, so an
# in-place upgrade from an older checkout needs them cleared out first.
pip uninstall -y substrate-interface scalecodec cyscale || true
pip install -e .

task validator

# Update observability server - redeploy if configs changed
if git diff HEAD~1 HEAD --name-only 2>/dev/null | grep -qE "grafana-training|loki-training|observability-server|vector/vector.toml"; then
    echo "Observability configs changed, redeploying observability server..."
    task deploy-observability-server
fi
