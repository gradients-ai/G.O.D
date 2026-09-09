# Validator Database

Database connection code, schema migrations, table/column constants, and SQL access modules.

## Contents

- `constants.py`: table names, column names, and SQL constants.
- `database.py`: Postgres connection/pool wrapper.
- `migrations/`: dbmate migration files.
- `sql/`: query modules grouped by domain.

Image prediction evaluation persists its case/configuration fingerprint in the
existing `task_nodes.eval_set_fingerprint` column, including outside boss rounds.
Raw-loss reads retain it so final ranking can reject mixed legacy/new metrics or
incompatible test cases. No schema migration is required for this change.
