# Validator scoring tests

Tests for `validator/scoring`: task scoring, compression ranking, the emission split and tournament weights. The rules they cover are in the [scoring README](../../../validator/scoring/README.md).

Run them from the repository root:

```bash
pytest tests/validator/scoring
```

## Contents

**Emission and weights**

| Path | Covers |
| --- | --- |
| `test_emission_balance.py` | The participation-driven allocator of the floating share. |
| `test_quant_emission_split.py` | The three-way split: minimums, caps, a tournament with no completed run, compression payouts to nodes, and the entry fees. |
| `test_tournament_pools.py` | Each type's pool is its dynamic base share, a burn-hotkey win adds nothing, and the pools and what they pay add up to 1.0. |
| `test_boss_rank_shares.py` | Paid places when the boss keeps the crown, through rank weights and node weights. |
| `test_boss_round_non_finite_scores.py` | A failed-but-scored compression entry has an infinite loss and reports no boss-round performance difference. |
| `test_tournament_scoring_pipeline.py` | Pairwise outcomes to points to weights, boss-round winner determination, and boss-round pair weights with third place. |
| `test_rank_weighted_scoring.py` | Rank-normalised score combination and pairwise results to win rates. |

**Compression scoring**

| Path | Covers |
| --- | --- |
| `test_quant_ranking.py` | Rank keys, the freeform cap, task scoring and SKU normalisation. |
| `test_quant_finalize.py` | Finalisation: each row's cap on the reference of its SKU, missing references, the bytes penalty, results rebuilt from stored rows, scored fails. |
| `test_quant_reference_eval.py` | Freeform scoring against the bf16 reference on each row's SKU. |
| `test_quant_eval_wiring.py` | Dispatch, the cached same-deployment reference, persisted scored fails, bytes. |
| `test_quant_chained_bar.py` | The common cap on a chained task's shared input, upstream-failed miners as scored fails, and the `quant_failure` write at finalisation. |
| `test_quant_structural_precheck.py` | The fixed-track structural check before dispatch, and authenticated hub reads. |

**Results and scorecard**

| Path | Covers |
| --- | --- |
| `test_infra_errors_not_scored.py` | Infrastructure failures on a compression task are re-queued, never scored. Scored fails stay scored. |
| `test_refinalize_reranks.py` | Re-finalising a task re-ranks every valid miner. |
| `test_image_score_validity.py` | An image entry that is not a fine-tune, has empty loss vectors or failed evaluation holds no valid score through persist, finalise and ranking. |
| `test_image_per_image_losses.py` | The per-image loss vector of an image tournament task: computed, persisted beside the scalar, absent for organic tasks and mismatched lengths. |
| `test_scorecard.py` | The freeform scorecard builder and the per-task assembly. |
| `test_scorecard_endpoint.py` | Non-finite scorecard numbers reach the auditing route as JSON null. |
| `__init__.py` | Package marker. |
