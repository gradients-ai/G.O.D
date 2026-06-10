from validator.infrastructure.repo_dedup import render_report
from validator.tournament.models import DedupResult
from validator.tournament.models import DedupTier
from validator.tournament.models import DupRelationship
from validator.tournament.models import PairVerdict


def _dedup_result() -> DedupResult:
    return DedupResult(
        cohort=["hotkey-a", "hotkey-b", "hotkey-c"],
        pair_verdicts=[
            PairVerdict(
                hotkey_a="hotkey-a",
                hotkey_b="hotkey-b",
                tier=DedupTier.T2,
                relationship=DupRelationship.DISTINCT,
                confidence=0.91,
                reason="Repository B adds a private checkpoint-selection strategy.",
            ),
            PairVerdict(
                hotkey_a="hotkey-a",
                hotkey_b="hotkey-c",
                tier=DedupTier.T1,
                relationship=DupRelationship.DUPLICATE,
                confidence=1.0,
                reason="Normalized sources are identical.",
            ),
        ],
        flagged_hotkeys=["hotkey-c"],
    )


def test_render_report_omits_distinct_verdicts_by_default():
    report = render_report(_dedup_result(), "tournament-1", "round-1", boss_hotkey=None)

    assert "## Flagged pairwise verdicts" in report
    assert "1 distinct pair(s) omitted from the published report" in report
    assert "private checkpoint-selection strategy" not in report
    assert "Normalized sources are identical." in report


def test_render_report_can_include_distinct_verdicts_for_dryrun():
    report = render_report(
        _dedup_result(),
        "tournament-1",
        "round-1",
        boss_hotkey=None,
        include_distinct_verdicts=True,
    )

    assert "## All pairwise verdicts" in report
    assert "private checkpoint-selection strategy" in report
