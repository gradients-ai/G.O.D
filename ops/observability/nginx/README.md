# Observability nginx

`nginx.conf` runs as `loki-gateway`, the only way in to the training Loki (port `3101`):

- `POST /loki/api/v1/push` needs the trainers' `LOKI_USERNAME`/`LOKI_PASSWORD` (`.htpasswd`, written by `task deploy-observability-server` from `.vali.env`).
- The public may only `GET` the query endpoints: `query`, `query_range`, `labels`, `label/<name>/values`, `series`, `index/stats` and `tail`.
- Everything else returns `403`.

`.htpasswd` is generated and git-ignored.
