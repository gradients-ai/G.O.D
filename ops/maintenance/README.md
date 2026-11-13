# Maintenance

Host-level maintenance scripts for operator machines. They modify system state: read a script before running it.

## Contents

| Path | Purpose |
| --- | --- |
| `move_docker_to_ephemeral.sh` | Prunes existing Docker data and relocates Docker storage to the ephemeral disk. |
