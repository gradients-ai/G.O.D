#!/bin/bash

set -e

echo "WARNING: This script will ERASE all current Docker data (containers, images, volumes)."
read -p "Are you sure you want to continue? Type 'Y' to proceed: " confirm

if [[ "$confirm" != "Y" ]]; then
  echo "Aborted."
  exit 1
fi

echo "Stopping Docker..."
sudo systemctl stop docker

echo "Removing all Docker containers..."
sudo docker container prune -f || true

echo "Removing all Docker images..."
sudo docker image prune -a -f || true

echo "Removing all Docker volumes..."
sudo docker volume prune -f || true

echo "Deleting Docker data directory at /var/lib/docker..."
sudo rm -rf /var/lib/docker

echo "Configuring daemon.json..."
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  },
  "data-root": "/ephemeral/docker"
}
EOF

echo "Creating new Docker root directory at /ephemeral/docker..."
sudo mkdir -p /ephemeral/docker
sudo chown root:root /ephemeral/docker
sudo chmod 711 /ephemeral/docker

echo "Starting Docker..."
sudo systemctl start docker

echo "Docker Root Dir now set to:"
docker info | grep "Docker Root Dir"
