#!/bin/bash
# Install and start NVIDIA DCGM for the trainer GPU monitor.
# Idempotent: a running hostengine returns immediately. A failed install does not
# stop the trainer; the monitor falls back to NVML until dcgmi answers.
# Run: bash ops/trainer_ops/install_dcgm.sh

set -euo pipefail

if [[ "${GPU_MONITOR_DCGM:-}" == "0" ]]; then
  echo "GPU_MONITOR_DCGM=0, leaving DCGM uninstalled"
  exit 0
fi

if [[ "$(id -u)" -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

if command -v dcgmi >/dev/null 2>&1 && systemctl is-active --quiet nvidia-dcgm; then
  exit 0
fi

if command -v dcgmi >/dev/null 2>&1 && systemctl cat nvidia-dcgm.service >/dev/null 2>&1; then
  systemctl enable nvidia-dcgm >/dev/null 2>&1 || true
  if systemctl start nvidia-dcgm; then
    exit 0
  fi
  echo "nvidia-dcgm did not start; the trainer will use NVML health checks until it does" >&2
  exit 0
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get is not available, so DCGM was not installed" >&2
  exit 1
fi

. /etc/os-release
distro="${ID}${VERSION_ID//./}"

if ! apt-cache show datacenter-gpu-manager >/dev/null 2>&1; then
  keyring="$(mktemp --suffix=.deb)"
  wget -q -O "$keyring" "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb"
  dpkg -i "$keyring"
  rm -f "$keyring"
  apt-get update -qq
fi

apt-get install -y datacenter-gpu-manager
systemctl enable nvidia-dcgm >/dev/null 2>&1 || true
if ! systemctl start nvidia-dcgm; then
  echo "datacenter-gpu-manager installed, but nvidia-dcgm did not start; the trainer will use NVML health checks until it does" >&2
fi
