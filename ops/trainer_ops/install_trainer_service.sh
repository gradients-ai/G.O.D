#!/bin/bash

# Script to install and enable the trainer.service for system startup
# This script must be run as root
# Run command: sudo bash ops/trainer_ops/install_trainer_service.sh

set -e

SERVICE_NAME="trainer.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="${SCRIPT_DIR}/${SERVICE_NAME}"
SYSTEMD_DIR="/etc/systemd/system"
TARGET_SERVICE_FILE="${SYSTEMD_DIR}/${SERVICE_NAME}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Error: This script must be run as root"
    echo "Please run with: sudo $0"
    exit 1
fi

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: Service file not found at ${SERVICE_FILE}"
    exit 1
fi

echo "Installing ${SERVICE_NAME}..."

# Copy service file to systemd directory
echo "Copying service file to ${SYSTEMD_DIR}..."
cp "$SERVICE_FILE" "$TARGET_SERVICE_FILE"
chmod 644 "$TARGET_SERVICE_FILE"

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable the service for startup
echo "Enabling ${SERVICE_NAME} for startup..."
systemctl enable "$SERVICE_NAME"

# Start the service
echo "Starting ${SERVICE_NAME}..."
systemctl restart "$SERVICE_NAME"

echo "Successfully installed, enabled, and started ${SERVICE_NAME}!"
echo ""
echo "Service status:"
systemctl status "$SERVICE_NAME" --no-pager -l || true
