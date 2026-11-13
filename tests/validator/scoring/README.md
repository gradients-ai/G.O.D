# Validator Scoring Tests

Tests for scoring and tournament scoring pipelines.

## Contents

- `test_quant_reference_eval.py`: freeform quant scoring against the bf16 reference: only decoded repos are evaluated, each with its reference model, and every row's bar is the reference on its own SKU.
- `test_quant_finalize.py`: quant finalize per-row bar on each row's SKU reference, reference exclusion, missing reference, same-SKU duplicates and scored-fail carriers.
- `test_quant_eval_wiring.py`: quant dispatch from `_evaluate_submissions` (reference model and cached SKUs), caching the same-deployment reference (first writer wins, mismatch alert), persisted scored fails, bytes, failed list.
- `test_quant_ranking.py`: quant rank keys, freeform bar, quant task scoring, and SKU normalisation (a uniform per-SKU drift leaves ranking unchanged).
- `test_quant_structural_precheck.py`: fixed-track structural check before dispatch (scored fail vs retried error),
  freeform never checked, token-authenticated `repo_info` and artifact byte reads.
- `test_scorecard.py`: the freeform scorecard: the pure builder and the per-task assembly (scored, scored fail, unscored, legacy, completed or not).
- `test_scorecard_endpoint.py`: the freeform scorecard through the real `/auditing/tasks/{id}` route: NaN and infinite floats are null, never a 500.
- `test_quant_chained_bar.py`: the chained freeform task's common bar on the shared input (upstream-failed players
  are scored fails), and the finalize-time `quant_failure` write (COALESCE, never NULLed).
- `test_quant_emission_split.py`: the three-way emission split (text 0.15 / quant 0.35 / image 0.225 minimums plus the 0.275 flex pool), the minimums applying with no completed quant tournament (no payee, the quant share left undistributed), and the per-type caps.
- `test_emission_balance.py`: the participation-driven flex-pool allocator over text, quant and image.
- `test_infra_errors_not_scored.py`: no infrastructure failure is scored as the miner's: `process_miners_pool` re-queues it, scored fails stay scored.
- `test_tournament_scoring_pipeline.py`: tournament scoring pipeline coverage.
- `__init__.py`: package marker.
