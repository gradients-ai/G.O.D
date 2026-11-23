"""Text seeds of rounds 1 and 2 never share a model: each draw skips the other live seed's model (by name without the org,
case-insensitive), on the first draw and on every redraw, and a pool holding nothing else falls back to it."""

import asyncio

import pytest

from tests.validator.tournament.test_seed_tasks import TID
from tests.validator.tournament.test_seed_tasks import _instruct
from tests.validator.tournament.test_seed_tasks import _live
from tests.validator.tournament.test_seed_tasks import env  # noqa: F401  (fixture)
from tests.validator.tournament.test_text_seed_rounds import TextWorld
from validator.tournament import constants as t_cst
from validator.tournament import seed_tasks
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentType


R1_BAND = t_cst.TEXT_R1_SIZE_B
R2_BAND = t_cst.TEXT_R2_SIZE_B


@pytest.fixture
def world(env, monkeypatch) -> TextWorld:  # noqa: F811
    return TextWorld(env, monkeypatch)


@pytest.fixture
def pools(world, monkeypatch) -> dict[tuple[float, float], list[str]]:
    """The model pool of each size band (repeated forever, like the content service), drawn by the fake instruct creator."""
    by_band: dict[tuple[float, float], list[str]] = {}

    async def text_models(_keypair, low, high):
        while True:
            for model_id in by_band.get((low, high), ["org/default-model"]):
                yield model_id

    async def make_instruct(config, models, datasets, **kwargs):
        task = _instruct(model_id=await anext(models))
        world.env.tasks[task.task_id] = task
        return task

    monkeypatch.setattr(seed_tasks, "_get_text_models", text_models)
    monkeypatch.setattr(seed_tasks, "create_synthetic_instruct_text_task", make_instruct)
    return by_band


def _model(world, role: SeedRole) -> str:
    return world.env.tasks[_live(world.env)[role].task_id].model_id


async def _create(world) -> None:
    await asyncio.wait_for(seed_tasks.create_seed_tasks(TID, world.env.config, TournamentType.TEXT), timeout=5)


async def _replace(world, role: SeedRole) -> None:
    new_seed = await asyncio.wait_for(
        seed_tasks.replace_seed(_live(world.env)[role], "task ended failure", world.env.config), timeout=5
    )
    assert new_seed is not None and new_seed.attempt == 2


class TestModelIdentity:
    def test_the_org_and_case_are_ignored(self):
        assert seed_tasks.model_identity("unsloth/Qwen2.5-1.5B-Instruct") == seed_tasks.model_identity(
            "Qwen/qwen2.5-1.5b-instruct"
        )
        assert seed_tasks.model_identity("Qwen2.5-1.5B-Instruct") == "qwen2.5-1.5b-instruct"

    def test_a_yarn_extended_copy_is_the_model_it_was_made_from(self):
        assert seed_tasks.model_identity("gradients-io/Qwen2.5-1.5B-Instruct-YaRN-extended-4") == seed_tasks.model_identity(
            "Qwen/Qwen2.5-1.5B-Instruct"
        )
        assert seed_tasks.model_identity("org/YaRN-extended-4-notes") == "yarn-extended-4-notes"

    def test_variants_with_another_name_are_different_models(self):
        names = {"Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-Math-1.5B"}
        assert len({seed_tasks.model_identity(name) for name in names}) == 3


class TestDistinctRoundSeedModels:
    async def test_the_first_r2_draw_skips_the_r1_model_under_another_org(self, world, pools):
        pools[R1_BAND] = ["Qwen/Qwen2.5-1.5B-Instruct"]
        pools[R2_BAND] = ["unsloth/qwen2.5-1.5b-instruct", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B-Instruct"]

        await _create(world)

        assert _model(world, SeedRole.TEXT_R1) == "Qwen/Qwen2.5-1.5B-Instruct"
        # The base model is a different model from the instruct one, so it is not skipped.
        assert _model(world, SeedRole.TEXT_R2) == "Qwen/Qwen2.5-1.5B"

    async def test_an_r2_redraw_skips_the_live_r1_model(self, world, pools):
        pools[R1_BAND] = ["org/model-a"]
        pools[R2_BAND] = ["org/model-b"]
        await _create(world)
        pools[R2_BAND] = ["mirror/MODEL-A", "org/model-c"]

        await _replace(world, SeedRole.TEXT_R2)

        assert _model(world, SeedRole.TEXT_R1) == "org/model-a"
        assert _model(world, SeedRole.TEXT_R2) == "org/model-c"

    async def test_an_r1_redraw_skips_the_live_r2_model(self, world, pools):
        pools[R1_BAND] = ["org/model-a"]
        pools[R2_BAND] = ["org/model-b"]
        await _create(world)
        pools[R1_BAND] = ["Mirror/Model-B", "org/model-c"]

        await _replace(world, SeedRole.TEXT_R1)

        assert _model(world, SeedRole.TEXT_R1) == "org/model-c"
        assert _model(world, SeedRole.TEXT_R2) == "org/model-b"

    async def test_a_replaced_seed_model_is_not_avoided(self, world, pools):
        pools[R1_BAND] = ["org/model-a"]
        pools[R2_BAND] = ["org/model-b"]
        await _create(world)
        pools[R1_BAND] = ["org/model-c"]
        await _replace(world, SeedRole.TEXT_R1)
        pools[R2_BAND] = ["org/model-a"]

        await _replace(world, SeedRole.TEXT_R2)

        replaced = [row for row in world.env.store.rows if row.status == SeedStatus.REPLACED]
        assert {row.role for row in replaced} == {SeedRole.TEXT_R1, SeedRole.TEXT_R2}
        assert _model(world, SeedRole.TEXT_R2) == "org/model-a"

    async def test_other_text_roles_draw_unconstrained(self, world, pools):
        pools[R1_BAND] = ["org/model-a"]
        pools[R2_BAND] = ["org/model-b"]
        pools[t_cst.SEED_INPUT_LARGE_SIZE_B] = ["org/model-a"]

        await _create(world)

        assert _model(world, SeedRole.TEXT_R3) == "org/model-a"

    async def test_a_pool_of_only_the_other_model_falls_back_to_it(self, world, pools, monkeypatch):
        monkeypatch.setattr(t_cst, "TEXT_SEED_DISTINCT_MODEL_MAX_SKIPS", 3)
        pools[R1_BAND] = ["org/model-a"]
        pools[R2_BAND] = ["org/model-a", "other/MODEL-A"]

        await _create(world)

        assert _model(world, SeedRole.TEXT_R1) == "org/model-a"
        assert seed_tasks.model_identity(_model(world, SeedRole.TEXT_R2)) == "model-a"


class TestOrderWithTheCompatibilityCheck:
    """The other seed's model is dropped before the quant-compatibility stage, and only skipped pool models count towards
    TEXT_SEED_DISTINCT_MODEL_MAX_SKIPS (a model that stage rejects does not)."""

    @pytest.fixture
    def checked(self, monkeypatch) -> list[str]:
        """Every model the compatibility stage was handed; it rejects the `incompatible-*` ones, like filter_quant_compatible."""
        seen: list[str] = []

        async def rejecting(models):
            async for model_id in models:
                seen.append(model_id)
                if "incompatible" not in model_id:
                    yield model_id

        monkeypatch.setattr(seed_tasks, "filter_quant_compatible", rejecting)
        return seen

    R2_POOL = [
        "mirror/MODEL-A",
        "org/incompatible-1",
        "mirror/MODEL-A",
        "org/incompatible-2",
        "mirror/MODEL-A",
        "org/model-c",
    ]

    async def test_the_other_model_never_reaches_the_compatibility_check(self, world, pools, checked):
        pools[R1_BAND] = ["org/model-a"]
        pools[R2_BAND] = self.R2_POOL
        await _create(world)
        del checked[:]
        await _replace(world, SeedRole.TEXT_R2)

        assert checked == ["org/incompatible-1", "org/incompatible-2", "org/model-c"]
        assert _model(world, SeedRole.TEXT_R2) == "org/model-c"

    async def test_past_the_skip_limit_the_other_model_is_checked_and_taken(self, world, pools, checked, monkeypatch):
        monkeypatch.setattr(t_cst, "TEXT_SEED_DISTINCT_MODEL_MAX_SKIPS", 2)
        pools[R1_BAND] = ["org/model-a"]
        pools[R2_BAND] = ["org/model-b"]
        await _create(world)
        pools[R2_BAND] = self.R2_POOL
        del checked[:]

        await _replace(world, SeedRole.TEXT_R2)

        # Two skips (the incompatible models in between are not counted), then the third occurrence goes through.
        assert checked == ["org/incompatible-1", "org/incompatible-2", "mirror/MODEL-A"]
        assert _model(world, SeedRole.TEXT_R2) == "mirror/MODEL-A"
