# End-to-end tests

End-to-end scripts and tests that exercise a whole flow.

## Contents

| Path | Purpose |
| --- | --- |
| `test_freeform_exposure_e2e.py` | Freeform exposure from decoder snapshot to public release and third-party verification. |
| `run_pvp_container.py` | Builds an image from `ops/docker/pvp-eval.dockerfile`, runs it with a config and validates the output JSON. |
| `run_pvp_suite.sh` | Runs a `pvp-eval:test` image over a set of configs within a time budget. |
| `pvp_remote_setup.sh` | Copies the repository to a remote GPU machine and builds `pvp-eval:test` there. |
| `run_pvp_remote.sh` | Runs `pvp-eval:test` on a machine prepared by `pvp_remote_setup.sh`. |

## Freeform exposure test

The test runs on CPU with no network.

| Real code | Faked |
| --- | --- |
| Decoder snapshot, output check, trainer decode job, decode recording | Object storage, the Hugging Face Hub |
| Scorecard and auditing endpoint, boss-round decoded publication | The Docker sandbox, the SQL layer |
| Decoder release and the verifier | The GitHub push |

```bash
OMP_NUM_THREADS=4 python -m pytest -q -o addopts='' tests/e2e/test_freeform_exposure_e2e.py
```
