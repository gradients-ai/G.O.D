# Core Constants

Shared constants that are safe for multiple runtimes to import.

## Contents

- `credentials.py`: environment-variable names and credential-related constants.
- `datasets.py`: image/text pair limits and dataset split constants.
- `docker.py`: shared Docker image names and container defaults.
- `environments.py`: supported environment names, environment images, and environment runtime configuration.
- `network.py`: subnet, chain, and network-level constants.
- `paths.py`: shared container/cache path constants.
- `training.py`: training defaults and task-type training constants.
- `__init__.py`: aggregate import surface for core constants.

Add new constants to the narrowest file that owns the domain. Avoid turning `__init__.py` into the only place a constant exists.
