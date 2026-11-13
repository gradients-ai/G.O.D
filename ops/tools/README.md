# Ops tools

Python and shell tools for operators, grouped by domain. Run Python tools from the repository root as modules, for example `python -m ops.tools.god_monitor`.

## Contents

| Path | Purpose |
| --- | --- |
| `config/` | Interactive env-file generation. |
| `datasets/` | Dataset generation tools. |
| `db/` | SQL snippets for inspecting task state. |
| `evaluation/` | Manual evaluation and reward function tools. |
| `god_monitor/` | Read-only tournament monitoring CLI with an interactive menu. |
| `observability/` | Log shipping smoke test. |
| `quant/` | Freeform publication tools: `reconcile_freeform_publications.py` and `verify_freeform_release.py`. |
| `scoring/` | Scoring and weight calculation inspection. |
| `simulations/` | Scoring and training-hours simulations. |
| `tasks/` | Task preparation probe. |
| `tournament/` | Tournament control and inspection tools. |
| `trainer/` | Trainer smoke test. |

## Freeform publication tools

`quant/reconcile_freeform_publications.py` inspects and re-runs the two public freeform publications: the champion's decoder release and the boss-round decoded model repos. The validator runs the same reconcile in the background. It needs the validator's `.vali.env`.

```bash
python -m ops.tools.quant.reconcile_freeform_publications                  # report only
python -m ops.tools.quant.reconcile_freeform_publications --apply          # reconcile and wait
python -m ops.tools.quant.reconcile_freeform_publications --release --tournament-id T --hotkey H [--allow-boss]
python -m ops.tools.quant.reconcile_freeform_publications --publish-decoded --tournament-id T
```

| Flag | Effect |
| --- | --- |
| none | Prints what a reconcile would publish. Writes nothing. |
| `--apply` | Runs the reconcile. |
| `--release` | Releases one decoder snapshot. `--allow-boss` admits the burn hotkey. |
| `--publish-decoded` | Publishes one tournament's boss-round decoded models. |
| `--json` | Machine-readable output. |

> [!IMPORTANT]
> `--apply`, `--release` and `--publish-decoded` write to public Hugging Face repositories. The tool takes no cross-process lock, so avoid running it while the validator is publishing the same item.

`quant/verify_freeform_release.py` verifies a released freeform result from public inputs only: the scorecard, the released decoder snapshot, the released repository and the artifact.

## See also

- [Freeform publications](../../validator/tournament/README.md#freeform-publications): what is published and when.
- [Evaluations and manual tools](../../docs/developer.md#evaluations-and-manual-tools): manual evaluation commands.
