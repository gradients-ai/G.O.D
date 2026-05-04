from types import SimpleNamespace

import pytest

from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import AugmentationScope
from core.models.model_prep_models import AugmentationType
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentStatus
from core.models.tournament_models import TournamentType
from core.models.utility_models import TaskType


class FakeStats:
    training = SimpleNamespace(init_loss=1.0, output_entropy=2.0)

    def model_dump(self):
        return {"training": {"init_loss": 1.0, "output_entropy": 2.0}}


def _entrypoint_args(**overrides):
    values = {
        "model": "base/model",
        "training_data": "/tmp/data.json",
        "task_type": TaskType.INSTRUCTTEXTTASK.value,
        "aug_type": None,
        "scope": None,
        "seed": None,
        "intensity": None,
        "source_model_repo": None,
        "source_task_id": None,
        "source_base_model_id": None,
        "reward_functions": None,
        "environment_name": None,
        "env_server_url": None,
        "num_episodes": 50,
        "task_id_min": 0,
        "task_id_max": 99999999,
        "env_payload_extra": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_previous_tournament_config_returns_none_without_completed_tournament(monkeypatch):
    from validator.utils import augmentation_decision

    async def no_tournament(*args, **kwargs):
        return None

    monkeypatch.setattr(augmentation_decision, "get_latest_completed_tournament", no_tournament)

    result = await augmentation_decision.maybe_get_boss_base_model_config(
        TaskType.INSTRUCTTEXTTASK,
        psql_db=object(),
    )

    assert result is None


@pytest.mark.asyncio
async def test_previous_tournament_config_returns_boss_repo_for_same_task_type(monkeypatch):
    from validator.utils import augmentation_decision

    tournament = TournamentData(
        tournament_id="tournament-1",
        tournament_type=TournamentType.TEXT,
        status=TournamentStatus.COMPLETED,
    )
    final_round = TournamentRoundData(
        round_id="round-1",
        tournament_id="tournament-1",
        round_number=3,
        round_type="knockout",
        is_final_round=True,
    )
    tournament_tasks = [
        SimpleNamespace(task_id="dpo-task"),
        SimpleNamespace(task_id="instruct-task"),
    ]
    source_tasks = {
        "dpo-task": SimpleNamespace(
            task_id="dpo-task",
            task_type=TaskType.DPOTASK,
            model_id="base/dpo-model",
        ),
        "instruct-task": SimpleNamespace(
            task_id="instruct-task",
            task_type=TaskType.INSTRUCTTEXTTASK,
            model_id="base/instruct-model",
        ),
    }

    async def get_latest(*args, **kwargs):
        return tournament

    async def get_rounds(*args, **kwargs):
        return [final_round]

    async def get_round_tasks(*args, **kwargs):
        return tournament_tasks

    async def get_task(task_id, *args, **kwargs):
        return source_tasks[str(task_id)]

    async def get_expected_repo(task_id, hotkey, *args, **kwargs):
        if str(task_id) == "instruct-task":
            return "expected-instruct-repo"
        return "expected-dpo-repo"

    monkeypatch.setattr(augmentation_decision, "get_latest_completed_tournament", get_latest)
    monkeypatch.setattr(augmentation_decision, "get_tournament_rounds", get_rounds)
    monkeypatch.setattr(augmentation_decision, "get_tournament_tasks", get_round_tasks)
    monkeypatch.setattr(augmentation_decision, "get_task", get_task)
    monkeypatch.setattr(augmentation_decision, "get_expected_repo_name", get_expected_repo)
    monkeypatch.setattr(augmentation_decision.random, "choice", lambda items: items[0])
    monkeypatch.setattr(augmentation_decision.random, "randint", lambda *args: 123)

    result = await augmentation_decision.maybe_get_boss_base_model_config(
        TaskType.INSTRUCTTEXTTASK,
        psql_db=object(),
    )

    assert result == AugmentationConfig(
        aug_type=AugmentationType.BOSS_BASE_MODEL,
        seed=123,
        source_model_repo="gradients-io-tournaments/expected-instruct-repo",
        source_task_id="instruct-task",
        source_base_model_id="base/instruct-model",
    )


def test_previous_tournament_augmentation_config_serializes():
    config = AugmentationConfig(
        aug_type=AugmentationType.BOSS_BASE_MODEL,
        seed=42,
        source_model_repo="boss/repo",
        source_task_id="task-id",
        source_base_model_id="base/model",
    )

    round_tripped = AugmentationConfig.model_validate(config.model_dump())

    assert round_tripped.aug_type == AugmentationType.BOSS_BASE_MODEL
    assert round_tripped.scope is None
    assert round_tripped.intensity is None
    assert round_tripped.source_model_repo == "boss/repo"


def test_previous_tournament_full_model_prep_uploads_source_repo(monkeypatch):
    from trainer.model_prep import boss_base_model

    calls = []
    config = AugmentationConfig(
        aug_type=AugmentationType.BOSS_BASE_MODEL,
        seed=7,
        source_model_repo="boss/full-model",
        source_task_id="task-id",
        source_base_model_id="base/model",
    )

    def fake_load(model_id, token):
        calls.append(("load", model_id))
        return "model", "tokenizer"

    def fake_upload(model, tokenizer, repo_id, token):
        calls.append(("upload", repo_id))

    monkeypatch.setattr(boss_base_model, "repo_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(boss_base_model, "check_for_lora", lambda *args, **kwargs: False)
    monkeypatch.setattr(boss_base_model, "load_model_and_tokenizer", fake_load)
    monkeypatch.setattr(boss_base_model, "upload_augmented_model", fake_upload)

    model, tokenizer = boss_base_model.prepare_boss_base_model(
        config,
        "anon/repo",
        "hf-token",
    )

    assert (model, tokenizer) == ("model", "tokenizer")
    assert calls == [("load", "boss/full-model"), ("upload", "anon/repo")]


def test_previous_tournament_lora_prep_merges_before_upload(monkeypatch):
    from trainer.model_prep import boss_base_model

    calls = []
    config = AugmentationConfig(
        aug_type=AugmentationType.BOSS_BASE_MODEL,
        seed=7,
        source_model_repo="boss/lora",
        source_task_id="task-id",
        source_base_model_id="base/model",
    )

    def fake_download_base(repo):
        calls.append(("download_base", repo))
        return "/base"

    def fake_download_lora(repo, local_dir):
        calls.append(("download_lora", repo, local_dir))
        return local_dir

    def fake_merge(base, lora, output):
        calls.append(("merge", base, lora, output))
        return "/merged"

    def fake_load(model_id, token):
        calls.append(("load", model_id))
        return "model", "tokenizer"

    def fake_upload(model, tokenizer, repo_id, token):
        calls.append(("upload", repo_id))

    monkeypatch.setattr(boss_base_model, "repo_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(boss_base_model, "check_for_lora", lambda *args, **kwargs: True)
    monkeypatch.setattr(boss_base_model, "_download_model_with_retry", fake_download_base)
    monkeypatch.setattr(boss_base_model, "_download_lora_with_retry", fake_download_lora)
    monkeypatch.setattr(boss_base_model, "_merge_base_and_lora", fake_merge)
    monkeypatch.setattr(boss_base_model, "load_model_and_tokenizer", fake_load)
    monkeypatch.setattr(boss_base_model, "upload_augmented_model", fake_upload)

    boss_base_model.prepare_boss_base_model(
        config,
        "anon/repo",
        "hf-token",
    )

    assert calls == [
        ("download_base", "base/model"),
        ("download_lora", "boss/lora", "/tmp/model_prep_lora/task-id"),
        ("merge", "/base", "/tmp/model_prep_lora/task-id", "/tmp/model_prep_merged"),
        ("load", "/merged"),
        ("upload", "anon/repo"),
    ]


def test_boss_base_existing_repo_loads_prepared_repo(monkeypatch):
    from trainer.model_prep import boss_base_model

    calls = []
    config = AugmentationConfig(
        aug_type=AugmentationType.BOSS_BASE_MODEL,
        seed=7,
        source_model_repo="boss/full-model",
        source_task_id="task-id",
        source_base_model_id="base/model",
    )

    def fake_load(model_id, token):
        calls.append(("load", model_id))
        return "model", "tokenizer"

    monkeypatch.setattr(boss_base_model, "repo_exists", lambda *args, **kwargs: True)
    monkeypatch.setattr(boss_base_model, "load_model_and_tokenizer", fake_load)

    boss_base_model.prepare_boss_base_model(config, "anon/repo", "hf-token")

    assert calls == [("load", "anon/repo")]


def test_entrypoint_no_augmentation_loads_original_and_computes_stats(monkeypatch):
    from trainer.model_prep import entrypoint

    calls = []

    def fake_load(model_id, token):
        calls.append(("load", model_id))
        return "model", "tokenizer"

    def fake_stats(*args, **kwargs):
        calls.append(("stats", args[0], args[1]))
        return FakeStats()

    monkeypatch.setattr(entrypoint, "parse_args", lambda: _entrypoint_args())
    monkeypatch.setattr(entrypoint, "load_training_data", lambda path: [{"text": "hello"}])
    monkeypatch.setattr(entrypoint, "load_model_and_tokenizer", fake_load)
    monkeypatch.setattr(entrypoint, "compute_text_stats", fake_stats)

    entrypoint.main()

    assert calls == [
        ("load", "base/model"),
        ("stats", "model", "tokenizer"),
    ]


def test_entrypoint_regular_augmentation_uses_existing_prepared_repo(monkeypatch):
    from trainer.model_prep import entrypoint

    calls = []

    def fake_load(model_id, token):
        calls.append(("load", model_id))
        return "model", "tokenizer"

    monkeypatch.setattr(
        entrypoint,
        "parse_args",
        lambda: _entrypoint_args(
            aug_type=AugmentationType.GAUSSIAN_NOISE.value,
            scope=AugmentationScope.SINGLE_LAYER.value,
            seed=123,
            intensity=0.01,
        ),
    )
    monkeypatch.setattr(entrypoint, "generate_anonymous_repo_name", lambda model_id, seed: "anon/repo")
    monkeypatch.setattr(entrypoint, "repo_exists", lambda *args, **kwargs: True)
    monkeypatch.setattr(entrypoint, "load_training_data", lambda path: [{"text": "hello"}])
    monkeypatch.setattr(entrypoint, "load_model_and_tokenizer", fake_load)
    monkeypatch.setattr(entrypoint, "augment_model", lambda *args, **kwargs: calls.append(("augment",)))
    monkeypatch.setattr(entrypoint, "upload_augmented_model", lambda *args, **kwargs: calls.append(("upload",)))
    monkeypatch.setattr(entrypoint, "compute_text_stats", lambda *args, **kwargs: FakeStats())

    entrypoint.main()

    assert calls == [("load", "anon/repo")]


def test_entrypoint_regular_augmentation_loads_original_then_uploads(monkeypatch):
    from trainer.model_prep import entrypoint

    calls = []

    def fake_load(model_id, token):
        calls.append(("load", model_id))
        return "model", "tokenizer"

    def fake_augment(model, config):
        calls.append(("augment", model, config.aug_type))

    def fake_upload(model, tokenizer, repo_id, token):
        calls.append(("upload", repo_id))

    monkeypatch.setattr(
        entrypoint,
        "parse_args",
        lambda: _entrypoint_args(
            aug_type=AugmentationType.GAUSSIAN_NOISE.value,
            scope=AugmentationScope.SINGLE_LAYER.value,
            seed=123,
            intensity=0.01,
        ),
    )
    monkeypatch.setattr(entrypoint, "generate_anonymous_repo_name", lambda model_id, seed: "anon/repo")
    monkeypatch.setattr(entrypoint, "repo_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(entrypoint, "load_training_data", lambda path: [{"text": "hello"}])
    monkeypatch.setattr(entrypoint, "load_model_and_tokenizer", fake_load)
    monkeypatch.setattr(entrypoint, "augment_model", fake_augment)
    monkeypatch.setattr(entrypoint, "upload_augmented_model", fake_upload)
    monkeypatch.setattr(entrypoint, "compute_text_stats", lambda *args, **kwargs: FakeStats())

    entrypoint.main()

    assert calls == [
        ("load", "base/model"),
        ("augment", "model", AugmentationType.GAUSSIAN_NOISE),
        ("upload", "anon/repo"),
    ]


def test_entrypoint_boss_base_model_skips_regular_augmentation(monkeypatch):
    from trainer.model_prep import entrypoint

    calls = []

    monkeypatch.setattr(
        entrypoint,
        "parse_args",
        lambda: _entrypoint_args(
            aug_type=AugmentationType.BOSS_BASE_MODEL.value,
            seed=123,
            source_model_repo="boss/repo",
            source_task_id="task-id",
            source_base_model_id="base/model",
        ),
    )
    monkeypatch.setattr(entrypoint, "generate_anonymous_repo_name", lambda model_id, seed: "anon/repo")
    monkeypatch.setattr(entrypoint, "load_training_data", lambda path: [{"text": "hello"}])
    monkeypatch.setattr(
        entrypoint,
        "prepare_boss_base_model",
        lambda config, repo_id, token: calls.append(("boss", repo_id, config.source_model_repo)) or ("model", "tokenizer"),
    )
    monkeypatch.setattr(entrypoint, "augment_model", lambda *args, **kwargs: calls.append(("augment",)))
    monkeypatch.setattr(entrypoint, "upload_augmented_model", lambda *args, **kwargs: calls.append(("upload",)))
    monkeypatch.setattr(entrypoint, "compute_text_stats", lambda *args, **kwargs: FakeStats())

    entrypoint.main()

    assert calls == [("boss", "anon/repo", "boss/repo")]
