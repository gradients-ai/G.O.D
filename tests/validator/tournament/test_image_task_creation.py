"""Image tournament task creation: the planned structure (one task per round, four boss-round tasks) and the three-task round 1
structure, replacements, and the exact pair count of a synthetic image task."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from core.models.image_models import ImageModelType
from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tasks.synthetics import constants as synth_cst
from validator.tasks.synthetics import diffusion
from validator.tournament import constants as t_cst
from validator.tournament import task_creator
from validator.tournament.models import Group
from validator.tournament.models import GroupRound
from validator.tournament.models import KnockoutRound
from validator.tournament.models import TournamentTask


ROUND_ID = "tourn_round_001"
BOSS_ROUND_ID = "tourn_round_004"
FAMILIES = {1: ImageModelType.FLUX, 2: ImageModelType.IDEOGRAM4, 3: ImageModelType.KREA2}


class World:
    """In-memory task store and image drawer: the creator's database seams and `create_synthetic_image_task` are faked."""

    def __init__(self, monkeypatch, planned: bool):
        self.registered: list[TournamentTask] = []
        self.tasks: dict[str, SimpleNamespace] = {}
        self.draws: list[dict] = []
        monkeypatch.setattr(task_creator, "is_planned_image_tournament", AsyncMock(return_value=planned))
        monkeypatch.setattr(task_creator, "get_tournament_tasks", AsyncMock(side_effect=self._tournament_tasks))
        monkeypatch.setattr(task_creator, "add_tournament_tasks", AsyncMock(side_effect=self._register))
        monkeypatch.setattr(task_creator.task_sql, "get_task", AsyncMock(side_effect=self._get_task))
        monkeypatch.setattr(task_creator, "get_task_gpu_requirement", lambda task: "gpu")
        monkeypatch.setattr(task_creator, "_get_image_models", self._pool)
        monkeypatch.setattr(task_creator, "create_synthetic_image_task", self._draw)

    @staticmethod
    async def _pool(_keypair):
        """An endless pool, as the model generator is: a family can be drawn any number of times."""
        while True:
            for model_type in ImageModelType:
                yield SimpleNamespace(model_id=f"repo/{model_type.value}", model_type=model_type)

    async def _draw(self, _config, models, *, max_num_prompts=None, num_prompts=None):
        model = await anext(models)
        task = SimpleNamespace(
            task_id=f"task-{len(self.tasks) + 1}",
            task_type=TaskType.IMAGETASK,
            model_type=model.model_type,
            model_id=model.model_id,
        )
        self.tasks[task.task_id] = task
        self.draws.append({"model_type": model.model_type, "max_num_prompts": max_num_prompts, "num_prompts": num_prompts})
        return task

    async def _register(self, tournament_tasks, _psql_db):
        self.registered.extend(tournament_tasks)

    async def _tournament_tasks(self, round_id, _psql_db):
        return [row for row in self.registered if row.round_id == round_id]

    async def _get_task(self, task_id, _psql_db):
        return self.tasks.get(str(task_id))

    def families(self) -> list[ImageModelType]:
        return [draw["model_type"] for draw in self.draws]


def _config():
    return MagicMock()


def _round(round_number: int) -> GroupRound:
    return GroupRound(
        round_id=f"tourn_round_{round_number:03d}", round_number=round_number, groups=[Group(member_ids=["a", "b", "c"])]
    )


@pytest.mark.parametrize("round_number", [1, 2, 3])
async def test_planned_round_creates_one_task_of_its_family_at_the_top_size(monkeypatch, round_number):
    world = World(monkeypatch, planned=True)
    round_data = _round(round_number)

    created = await task_creator.create_image_tournament_tasks(round_data, "tourn", _config())

    assert len(created) == 1
    assert world.families() == [FAMILIES[round_number]]
    assert world.draws[0]["num_prompts"] == synth_cst.MAX_IMAGE_SYNTH_PAIRS == 50
    assert [(row.round_id, row.group_id, row.pair_id) for row in world.registered] == [
        (round_data.round_id, f"{round_data.round_id}_group_001", None)
    ]


async def test_planned_round_is_idempotent(monkeypatch):
    world = World(monkeypatch, planned=True)
    round_data = _round(2)

    first = await task_creator.create_image_tournament_tasks(round_data, "tourn", _config())
    second = await task_creator.create_image_tournament_tasks(round_data, "tourn", _config())

    assert second == first
    assert len(world.draws) == 1
    assert len(world.registered) == 1


async def test_a_round_outside_the_plan_is_refused(monkeypatch):
    World(monkeypatch, planned=True)

    with pytest.raises(ValueError, match="not a miner round"):
        await task_creator.create_image_tournament_tasks(_round(4), "tourn", _config())


def test_image_round_spec_matches_the_plan():
    assert [task_creator.image_round_spec(n) for n in (1, 2, 3)] == list(t_cst.IMAGE_ROUND_PLAN)
    with pytest.raises(ValueError):
        task_creator.image_round_spec(0)


async def test_planned_boss_round_creates_one_task_per_family_at_the_top_size(monkeypatch):
    world = World(monkeypatch, planned=True)

    created = await task_creator.create_image_tournament_tasks(
        KnockoutRound(round_id=BOSS_ROUND_ID, round_number=4, pairs=[("boss", "contender")]),
        "tourn",
        _config(),
        is_final_round=True,
    )

    assert len(created) == 4
    assert sorted(f.value for f in world.families()) == sorted(f.value for f in t_cst.IMAGE_BOSS_ROUND_TASK_DISTRIBUTION)
    assert {draw["num_prompts"] for draw in world.draws} == {50}
    assert {row.pair_id for row in world.registered} == {f"{BOSS_ROUND_ID}_pair_001"}
    assert all(row.group_id is None for row in world.registered)


async def test_planned_boss_round_rerun_creates_only_the_missing_families(monkeypatch):
    world = World(monkeypatch, planned=True)
    boss_round = KnockoutRound(round_id=BOSS_ROUND_ID, round_number=4, pairs=[("boss", "contender")])
    await task_creator.create_image_tournament_tasks(boss_round, "tourn", _config(), is_final_round=True)
    kept = world.registered[:2]
    kept_families = {world.tasks[row.task_id].model_type for row in kept}
    world.registered[:] = kept
    world.draws.clear()

    created = await task_creator.create_image_tournament_tasks(boss_round, "tourn", _config(), is_final_round=True)

    assert len(created) == 4
    assert set(world.families()) == set(t_cst.IMAGE_BOSS_ROUND_TASK_DISTRIBUTION) - kept_families
    assert len(world.draws) == 2
    assert len(world.registered) == 4


async def test_planned_boss_round_creates_nothing_when_complete(monkeypatch):
    world = World(monkeypatch, planned=True)
    boss_round = KnockoutRound(round_id=BOSS_ROUND_ID, round_number=4, pairs=[("boss", "contender")])
    await task_creator.create_image_tournament_tasks(boss_round, "tourn", _config(), is_final_round=True)
    world.draws.clear()

    await task_creator.create_image_tournament_tasks(boss_round, "tourn", _config(), is_final_round=True)

    assert world.draws == []


async def test_planned_boss_round_short_of_a_task_raises_and_the_next_pass_completes_it(monkeypatch):
    world = World(monkeypatch, planned=True)
    monkeypatch.setattr(task_creator.asyncio, "sleep", AsyncMock())
    boss_round = KnockoutRound(round_id=BOSS_ROUND_ID, round_number=4, pairs=[("boss", "contender")])
    draw = world._draw

    async def failing_qwen(config, models, **kwargs):
        model = await anext(models)
        if model.model_type == ImageModelType.QWEN_IMAGE:
            raise ValueError("generation failed")

        async def same_model():
            while True:
                yield model

        return await draw(config, same_model(), **kwargs)

    monkeypatch.setattr(task_creator, "create_synthetic_image_task", failing_qwen)
    with pytest.raises(RuntimeError, match="holds 3 of its 4 tasks"):
        await task_creator.create_image_tournament_tasks(boss_round, "tourn", _config(), is_final_round=True)
    assert len(world.registered) == 3

    monkeypatch.setattr(task_creator, "create_synthetic_image_task", draw)
    created = await task_creator.create_image_tournament_tasks(boss_round, "tourn", _config(), is_final_round=True)

    assert len(created) == 4
    assert world.families()[-1] == ImageModelType.QWEN_IMAGE


async def test_three_task_round_one_structure_round_one_has_three_tasks_with_the_pair_cap(monkeypatch):
    world = World(monkeypatch, planned=False)

    created = await task_creator.create_image_tournament_tasks(_round(1), "tourn", _config())

    assert len(created) == t_cst.SMALL_TOURNAMENT_GROUP_TASKS == 3
    assert world.families() == list(t_cst.ROUND_ONE_IMAGE_MODEL_TYPES)
    assert {draw["max_num_prompts"] for draw in world.draws} == {t_cst.ROUND_ONE_IMAGE_MAX_SYNTH_PAIRS} == {30}
    assert {draw["num_prompts"] for draw in world.draws} == {None}


async def test_three_task_round_one_structure_knockout_round_two_creates_one_task_per_pair(monkeypatch):
    world = World(monkeypatch, planned=False)

    created = await task_creator.create_image_tournament_tasks(
        KnockoutRound(round_id="tourn_round_002", round_number=2, pairs=[("a", "b")]), "tourn", _config()
    )

    assert len(created) == 1
    assert world.families()[0] in t_cst.ROUND_TWO_IMAGE_MODEL_TYPES
    assert world.draws[0]["num_prompts"] is None
    assert [(row.group_id, row.pair_id) for row in world.registered] == [(None, "tourn_round_002_pair_001")]


async def test_three_task_round_one_structure_boss_round_has_the_six_task_mix_with_random_sizes(monkeypatch):
    world = World(monkeypatch, planned=False)

    created = await task_creator.create_image_tournament_tasks(
        KnockoutRound(round_id=BOSS_ROUND_ID, round_number=4, pairs=[("boss", "contender")]),
        "tourn",
        _config(),
        is_final_round=True,
    )

    assert len(created) == sum(t_cst.FINAL_ROUND_IMAGE_TASK_DISTRIBUTION.values()) == 6
    for model_type, count in t_cst.FINAL_ROUND_IMAGE_TASK_DISTRIBUTION.items():
        assert world.families().count(model_type) == count
    assert {draw["num_prompts"] for draw in world.draws} == {None}


def _replacement_world(monkeypatch, planned: bool, original_nodes: list):
    original = SimpleNamespace(
        task_id="orig-task",
        task_type=TaskType.IMAGETASK,
        status="prep_task_failure",
        model_id="repo/krea2",
        model_type=ImageModelType.KREA2,
        model_params_count=0,
    )
    world = World(monkeypatch, planned=planned)
    world.tasks["orig-task"] = original
    monkeypatch.setattr(task_creator.seeds_sql, "is_seeded_text_tournament", AsyncMock(return_value=False))
    monkeypatch.setattr(task_creator.task_sql, "get_nodes_assigned_to_task", AsyncMock(return_value=original_nodes))
    monkeypatch.setattr(task_creator.task_sql, "get_expected_repo_name", AsyncMock(return_value="repo-name"))
    monkeypatch.setattr(task_creator.task_sql, "assign_node_to_task", AsyncMock())
    monkeypatch.setattr(task_creator.task_sql, "set_expected_repo_name", AsyncMock())
    monkeypatch.setattr(task_creator.task_sql, "detach_task_from_tournament", AsyncMock())
    return world


async def test_planned_round_replacement_keeps_the_family_the_size_and_the_boss(monkeypatch):
    nodes = [SimpleNamespace(hotkey=EMISSION_BURN_HOTKEY), SimpleNamespace(hotkey="miner-1"), SimpleNamespace(hotkey="miner-2")]
    world = _replacement_world(monkeypatch, planned=True, original_nodes=nodes)

    new_task_id = await task_creator.replace_tournament_task(
        "orig-task", "tourn", "tourn_round_003", "tourn_round_003_group_001", None, _config()
    )

    assert world.families() == [ImageModelType.KREA2]
    assert world.draws[0]["num_prompts"] == synth_cst.MAX_IMAGE_SYNTH_PAIRS
    assigned = [(call.args[0], call.args[1].hotkey) for call in task_creator.task_sql.assign_node_to_task.await_args_list]
    assert assigned == [(new_task_id, node.hotkey) for node in nodes]
    assert (new_task_id, EMISSION_BURN_HOTKEY) in assigned
    assert [row.group_id for row in world.registered] == ["tourn_round_003_group_001"]
    assert task_creator.task_sql.detach_task_from_tournament.await_args.args[:2] == ("orig-task", "tourn")


async def test_three_task_round_one_structure_replacement_keeps_the_random_size(monkeypatch):
    nodes = [SimpleNamespace(hotkey=EMISSION_BURN_HOTKEY), SimpleNamespace(hotkey="miner-1")]
    world = _replacement_world(monkeypatch, planned=False, original_nodes=nodes)

    await task_creator.replace_tournament_task("orig-task", "tourn", BOSS_ROUND_ID, None, f"{BOSS_ROUND_ID}_pair_001", _config())

    assert world.families() == [ImageModelType.KREA2]
    assert world.draws[0]["num_prompts"] is None
    assert task_creator.task_sql.assign_node_to_task.await_count == 2


@pytest.fixture
def synthetic_task_seams(monkeypatch):
    requested: list[int] = []

    async def generate(_config, num_prompts, _category):
        requested.append(num_prompts)
        return [SimpleNamespace(image_url=f"i{n}", text_url=f"t{n}") for n in range(num_prompts)], "prefix", None

    async def pool():
        yield SimpleNamespace(model_id="repo/flux", model_type=ImageModelType.FLUX)

    monkeypatch.setattr(diffusion, "generate_image_synthetic_by_category", generate)
    monkeypatch.setattr(diffusion, "pick_image_synth_category", lambda: "style")
    monkeypatch.setattr(diffusion, "maybe_get_augmentation_config", lambda task_type: None)
    monkeypatch.setattr(diffusion, "add_task", AsyncMock(side_effect=lambda task, psql_db: task))
    monkeypatch.setattr(diffusion, "ImageRawTask", lambda **fields: SimpleNamespace(**fields))
    monkeypatch.setattr(diffusion, "TEMP_PATH_FOR_IMAGES", "/tmp/imgv2-b2-unused")
    monkeypatch.setattr(diffusion.Path, "mkdir", lambda *a, **k: None)
    return SimpleNamespace(requested=requested, pool=pool)


@pytest.mark.parametrize("count", [synth_cst.MIN_IMAGE_SYNTH_PAIRS, 37, synth_cst.MAX_IMAGE_SYNTH_PAIRS])
async def test_num_prompts_requests_exactly_that_many_pairs(synthetic_task_seams, count):
    task = await diffusion.create_synthetic_image_task(_config(), synthetic_task_seams.pool(), num_prompts=count)

    assert synthetic_task_seams.requested == [count]
    assert len(task.image_text_pairs) == count


@pytest.mark.parametrize("count", [0, synth_cst.MIN_IMAGE_SYNTH_PAIRS - 1, synth_cst.MAX_IMAGE_SYNTH_PAIRS + 1])
async def test_num_prompts_out_of_range_is_rejected(synthetic_task_seams, count):
    with pytest.raises(ValueError, match="num_prompts"):
        await diffusion.create_synthetic_image_task(_config(), synthetic_task_seams.pool(), num_prompts=count)

    assert synthetic_task_seams.requested == []


async def test_without_num_prompts_the_size_is_drawn_up_to_the_cap(synthetic_task_seams):
    await diffusion.create_synthetic_image_task(_config(), synthetic_task_seams.pool(), max_num_prompts=12)

    assert synth_cst.MIN_IMAGE_SYNTH_PAIRS <= synthetic_task_seams.requested[0] <= 12
