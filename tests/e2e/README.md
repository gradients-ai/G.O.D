# E2E Tests

End-to-end scripts for PvP and remote evaluation flows.

## Contents

- `pvp_remote_setup.sh`: setup helper for remote PvP evaluation.
- `run_pvp_container.py`: run PvP evaluation inside a container.
- `run_pvp_remote.sh`: remote PvP evaluation wrapper.
- `run_pvp_suite.sh`: shell wrapper for the PvP suite.
- `test_freeform_exposure_e2e.py`: freeform exposure end to end (see below).
- `__init__.py`: package marker.

## Freeform exposure e2e

CPU only, no network, about 2 s. Real code for the decoder snapshot (git + xz), the upload check, the trainer decode job
(private work dir, permissions), the orchestrator's decode recording, hub-listing hashes, the scorecard and auditing
endpoint, boss-round decoded publication, the champion handover release and the third-party verifier
(`ops/tools/quant/verify_freeform_release.py`). Faked: MinIO, the HF hub, the docker sandbox (the decoder runs on the host
with the sandbox's argv/env), the SQL layer and the GitHub push.

```bash
OMP_NUM_THREADS=4 python -m pytest -q -o addopts='' -p no:randomly tests/e2e/test_freeform_exposure_e2e.py
```
