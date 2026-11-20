#!/bin/bash
# Put Caddy in front of the subnet API: HTTPS with automatic Let's Encrypt
# certificates for SUBNET_API_DOMAIN, proxied to 127.0.0.1:SUBNET_API_PORT.
#
# Runs from `task validator` (and so on every auto-update). It is a no-op when
# SUBNET_API_DOMAIN is unset, and only calls sudo when Caddy is missing or
# /etc/caddy/Caddyfile needs to change.
#
# Needs: DNS A/AAAA for SUBNET_API_DOMAIN pointing at this machine, and ports
# 80 and 443 open (Let's Encrypt validates over them).
set -euo pipefail

set -a
. ./.vali.env
set +a

if [ -z "${SUBNET_API_DOMAIN:-}" ]; then
    echo "SUBNET_API_DOMAIN not set; skipping Caddy setup"
    exit 0
fi
if [ -z "${SUBNET_API_PORT:-}" ]; then
    echo "SUBNET_API_DOMAIN is set but SUBNET_API_PORT is not; set both in .vali.env" >&2
    exit 1
fi

CADDYFILE=/etc/caddy/Caddyfile

if ! command -v caddy >/dev/null 2>&1; then
    echo "Installing Caddy from the official apt repository..."
    sudo apt-get update -qq
    sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl gnupg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
    sudo chmod o+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg /etc/apt/sources.list.d/caddy-stable.list
    sudo apt-get update -qq
    sudo apt-get install -y caddy
fi

desired=$(cat <<EOF
# Managed by G.O.D ops/validator_ops/setup_caddy.sh — edits are overwritten.
${SUBNET_API_DOMAIN} {
	encode zstd gzip
	reverse_proxy 127.0.0.1:${SUBNET_API_PORT}
}
EOF
)

changed=0
if [ -f "$CADDYFILE" ] && [ "$(cat "$CADDYFILE")" = "$desired" ]; then
    echo "Caddyfile up to date"
else
    changed=1
    tmp=$(mktemp)
    printf '%s\n' "$desired" > "$tmp"
    caddy validate --config "$tmp" --adapter caddyfile
    if [ -f "$CADDYFILE" ]; then
        backup="${CADDYFILE}.bak.$(date +%Y%m%d%H%M%S)"
        sudo cp "$CADDYFILE" "$backup"
        echo "Previous Caddyfile saved to $backup"
    fi
    sudo install -m 644 "$tmp" "$CADDYFILE"
    rm -f "$tmp"
    echo "Wrote $CADDYFILE for ${SUBNET_API_DOMAIN} -> 127.0.0.1:${SUBNET_API_PORT}"
fi

if ! systemctl is-active --quiet caddy; then
    sudo systemctl enable --now caddy
elif [ "$changed" = 1 ]; then
    sudo systemctl reload caddy
fi
echo "Caddy serving https://${SUBNET_API_DOMAIN}"
