from core.models.payload_models import PreparedTrainingImage
from core.models.payload_models import TrainerPrefetchEntry
from core.models.tournament_models import TournamentType
from validator.tournament.image_builds.batch import _variants
from validator.tournament.image_builds.depot import classify_failure
from validator.tournament.models import ImageBuildVariant


def test_text_tournament_has_one_build_variant():
    assert _variants(TournamentType.TEXT) == (ImageBuildVariant.TEXT,)


def test_image_tournament_has_both_runtime_variants():
    assert _variants(TournamentType.IMAGE) == (
        ImageBuildVariant.IMAGE_FLUX,
        ImageBuildVariant.IMAGE_TOOLKIT,
    )


def test_timeout_is_retryable_failure_class():
    assert classify_failure("cancelled", timed_out=True) == "timeout"


def test_bad_dockerfile_is_deterministic_failure_class():
    output = "failed to solve: process did not complete successfully: exit code: 1"
    assert classify_failure(output, timed_out=False) == "dockerfile"


def test_prefetch_contract_keeps_digest_pinned_identity():
    image = PreparedTrainingImage(
        build_id="build-1",
        pull_ref="registry.depot.dev/project:tag",
        digest="sha256:" + "a" * 64,
        variant="text",
    )
    entry = TrainerPrefetchEntry(**image.model_dump(), priority=7)
    assert entry.digest == image.digest
    assert entry.priority == 7
