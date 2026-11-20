#!/bin/sh
set -eu
wget -q -O /tmp/extra.bin https://example.com/extra.bin
echo "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  /tmp/extra.bin" | sha256sum --check
uv pip install --system -r /workspace/requirements.txt
