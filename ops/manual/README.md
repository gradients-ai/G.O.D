# Manual Ops

Manual probes, smoke tests, local examples, and debugging scripts. These are not part of the normal service startup path.

## Contents

- `examples/`: local task launch examples.
- `trainer/`: trainer-specific manual probes.
- `validator/`: validator-specific manual probes.
- `create_synthetic_env_task.py`: create a synthetic environment task manually.
- `dataset_whitelist_smoke.sh`: direct trainer request for whitelisted dataset download testing.
- `environment_tournament_flow_probe.py`: end-to-end environment tournament flow probe.
- `evaluation_flow_debugging.md`: notes for debugging evaluation flow.
- `example.http`: HTTP request examples.
- `move_docker_to_ephemeral.sh`: local Docker storage relocation helper.
- `__init__.py`: package marker.
