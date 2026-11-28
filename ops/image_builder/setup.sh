#!/usr/bin/env bash
# Set up or update the tournament image build box. Run as root from this directory, with .env filled in (env.example).
#
#   ./setup.sh                  certificate, registry users and config, BuildKit builder, Zot and Woodpecker (idempotent)
#   ./setup.sh renew            renew the certificate and restart the services that serve it (daily cron)
#   ./setup.sh configure-repo   after the first Woodpecker login: enable the snapshot repo, add its secret, push the recipe
#
# Secrets it generates stay in ./state (mode 700): registry passwords, agent secret, certificate.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
DIR=$(pwd)
STATE=$DIR/state
LEGO_VERSION=v5.7.0
BUILDKIT_IMAGE=moby/buildkit:v0.35.1

[ -f .env ] || { echo "missing .env (copy env.example)"; exit 1; }
set -a; . ./.env; set +a
: "${BUILD_BOX_HOST:?}"
mkdir -p "$STATE/certs"; chmod 700 "$STATE"

is_ip() { [[ $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; }

certificate() {
  if ! command -v lego >/dev/null; then
    curl -sSL "https://github.com/go-acme/lego/releases/download/$LEGO_VERSION/lego_${LEGO_VERSION}_linux_amd64.tar.gz" \
      | tar xz -C /usr/local/bin lego
  fi
  # A certificate for a bare IP is only issued with the short-lived profile (about 6 days); a name gets 90 days.
  local profile=(); is_ip "$BUILD_BOX_HOST" && profile=(--profile shortlived)
  lego run --accept-tos --http --domains "$BUILD_BOX_HOST" "${profile[@]}" --path "$STATE/lego" --log.format text >/dev/null
  cp "$STATE/lego/certificates/$BUILD_BOX_HOST.crt" "$STATE/certs/server.crt"
  cp "$STATE/lego/certificates/$BUILD_BOX_HOST.key" "$STATE/certs/server.key"
}

secrets() {
  for user in builder trainer; do
    [ -s "$STATE/$user.pass" ] || (umask 077; openssl rand -hex 24 > "$STATE/$user.pass")
  done
  [ -s "$STATE/agent.secret" ] || (umask 077; openssl rand -hex 32 > "$STATE/agent.secret")
  : > "$STATE/htpasswd"
  for user in builder trainer; do
    docker run --rm -i httpd:2.4-alpine htpasswd -niB "$user" < "$STATE/$user.pass" | grep . >> "$STATE/htpasswd"
  done
  chmod 644 "$STATE/htpasswd"
}

zot_config() {
  (umask 077; python3 - "$DIR/zot-config.template.json" > "$STATE/zot-config.json" <<'PY'
import json, os, re, sys
text = open(sys.argv[1]).read()
text = re.sub(r"\$\{(\w+)\}", lambda m: json.dumps(os.environ[m.group(1)])[1:-1], text)
json.loads(text)
print(text)
PY
  )
}

loopback() {
  # On a NATed box the public IP is not on any interface, so pushes to it would leave the box and come back. Answer it
  # locally instead (persisted with a systemd unit).
  local ip=$BUILD_BOX_HOST
  is_ip "$ip" || ip=$(getent ahostsv4 "$ip" | awk 'NR==1{print $1}')
  ip -o addr | grep -q " $ip/" && return 0
  cat > /etc/systemd/system/god-image-builder-loopback.service <<EOF
[Unit]
Description=Answer this box's public IP locally (god-image-builder)
After=network-online.target
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'ip addr add $ip/32 dev lo || true'
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload && systemctl enable --now god-image-builder-loopback.service >/dev/null
}

builder() {
  if ! docker buildx inspect god >/dev/null 2>&1; then
    docker buildx create --name god --driver docker-container --driver-opt "image=$BUILDKIT_IMAGE" \
      --driver-opt network=host --buildkitd-config "$DIR/buildkitd.toml" --bootstrap >/dev/null
  fi
  docker login "$BUILD_BOX_HOST:5000" -u builder --password-stdin < "$STATE/builder.pass" >/dev/null
}

cron() {
  echo "17 4 * * * root $DIR/setup.sh renew >> /var/log/god-image-builder-renew.log 2>&1" > /etc/cron.d/god-image-builder
}

start() {
  WOODPECKER_AGENT_SECRET=$(cat "$STATE/agent.secret") docker compose up -d
}

case "${1:-setup}" in
  setup)
    certificate; secrets; zot_config; loopback; start; builder; cron
    echo "Ready. Registry: https://$BUILD_BOX_HOST:5000 (users builder, trainer: $STATE/*.pass)"
    echo "Next: log in at https://$BUILD_BOX_HOST, create an API token, set WOODPECKER_TOKEN in .env, run ./setup.sh configure-repo"
    ;;
  renew)
    # lego renews only when due (about halfway through a short-lived certificate); restart only when it did.
    before=$(sha256sum "$STATE/certs/server.crt" 2>/dev/null || true)
    certificate
    [ "$before" = "$(sha256sum "$STATE/certs/server.crt")" ] || docker restart zot woodpecker-server >/dev/null
    ;;
  configure-repo)
    exec python3 "$DIR/configure_repo.py"
    ;;
  *)
    echo "usage: $0 [setup|renew|configure-repo]"; exit 2
    ;;
esac
