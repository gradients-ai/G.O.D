#!/usr/bin/env bash
# Build the four NL2Bash filesystem snapshots used by validator/evaluation/evaluators/intercode.py.
#
# Upstream intercode (princeton-nlp/intercode) ships one setup_nl2b_fs_{1,2,3,4}.sh per
# variant; each is intended to be baked into its own image. Here we run all four
# inside the same builder stage and capture their post-state as tarballs under
# /intercode_fs/, so the runtime image can restore any of them on demand without
# needing docker.
#
# Inputs (set by the dockerfile):
#   INTERCODE_REPO   path to a checkout of princeton-nlp/intercode
# Outputs:
#   /intercode_fs/fs1.tar, fs2.tar, fs3.tar   (fs4 has no managed filesystem)

set -uo pipefail

INTERCODE_REPO="${INTERCODE_REPO:-/opt/intercode}"
OUT_DIR="${OUT_DIR:-/intercode_fs}"
mkdir -p "$OUT_DIR"

declare -A FS_PATHS=(
    [1]="/testbed"
    [2]="/system"
    [3]="/workspace /backup"
)

# fs4 has no managed paths (filesystem-agnostic tasks).
for fs in 1 2 3; do
    echo "=== building intercode fs_${fs} ==="
    script="$INTERCODE_REPO/docker/bash_scripts/setup_nl2b_fs_${fs}.sh"
    if [[ ! -f "$script" ]]; then
        echo "missing setup script: $script" >&2
        exit 1
    fi
    chmod +x "$script"
    # Run setup; intercode's scripts don't use `set -e`, so individual failures
    # (e.g. the non-standard `mkdir -d` line in setup_nl2b_fs_3.sh) are tolerated
    # the same way they are upstream.
    bash "$script" || echo "warning: setup_nl2b_fs_${fs}.sh exited non-zero (matches upstream behavior)"

    # Snapshot the paths this variant manages.
    paths="${FS_PATHS[$fs]}"
    # tar paths relative to / so extraction with `tar -xpf ... -C /` restores them.
    tar_args=()
    for p in $paths; do
        if [[ -e "$p" ]]; then
            tar_args+=("$(realpath --relative-to=/ "$p")")
        else
            echo "warning: expected path $p not created by fs_${fs} setup" >&2
        fi
    done
    if (( ${#tar_args[@]} == 0 )); then
        echo "no paths to snapshot for fs_${fs}, skipping" >&2
        continue
    fi
    (cd / && tar --acls --xattrs -cpf "$OUT_DIR/fs${fs}.tar" "${tar_args[@]}")
    echo "wrote $OUT_DIR/fs${fs}.tar"

    # Clean up before the next variant so paths don't leak across snapshots.
    for p in /testbed /system /workspace /backup; do
        rm -rf "$p"
    done
done

# fs4 has no setup; record an empty tar so the eval script can treat all fs
# versions uniformly (it skips extraction when no managed paths are present,
# but having the file simplifies error checking).
tar -cf "$OUT_DIR/fs4.tar" -T /dev/null
echo "wrote $OUT_DIR/fs4.tar (empty, fs_4 is filesystem-agnostic)"