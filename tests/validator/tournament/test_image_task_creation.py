"""Image tournament task creation: the planned structure (one task per round, four boss-round tasks drawn around round 3's
slot) and the three-task round 1 structure, replacements, and the pair cap of a synthetic image task."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from core.models.image_models import ImageModelType
from core.models.image_models import ImageTrainingMode
from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tasks.synthetics import constants as synth_cst
from validator.tasks.synthetics import diffusion
from validator.tournament import constants as t_cst
from validator.tournament import task_creator
from validator.tournament.models import Group
from validator.tournament.models import GroupRound
from validator.tournament.models import KnockoutRound
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask


TID = "tourn"
BOSS_ROUND_ID = "tourn_round_004"
ROUND_THREE_ID = "tourn_round_003"
EDIT = ImageTrainingMode.EDIT
T2I = ImageTrainingMode.TEXT_TO_IMAGE


class World:
    """In-memory task store and image drawer: the creator's database seams and `create_synthetic_image_task` are faked."""

    def __init__(self, monkeypatch, planned: bool):
        self.registered: list[TournamentTask] = []
        self.rounds: list[TournamentRoundData] = []
        self.tasks: dict[str, SimpleNamespace] = {}
        self.draws: list[dict] = []
        monkeypatch.setattr(task_creator, "is_planned_image_tournament", AsyncMock(return_value=planned))
        monkeypatch.setattr(task_creator, "get_tournament_tasks", AsyncMock(side_effect=self._tournament_tasks))
        monkeypatch.setattr(task_creator, "get_tournament_rounds", AsyncMock(side_effect=self._tournament_rounds))
        monkeypatch.setattr(task_creator, "add_tournament_tasks", AsyncMock(side_effect=self._register))
        monkeypatch.setattr(task_creator.task_sql, "get_task", AsyncMock(side_effect=self._get_task))
        monkeypatch.setattr(task_creator, "get_task_gpu_requirement", lambda task: "gpu")
        monkeypatch.setattr(task_creator, "_get_image_models", self._pool)
        monkeypatch.setattr(task_creator, "create_synthetic_image_task", self._draw)
        monkeypatch.setattr(task_creator.asyncio, "sleep", AsyncMock())

    @staticmethod
    async def _pool(_keypair):
        """An endless pool, as the model generator is: a family can be drawn any number of times."""
        while True:
            for model_type in ImageModelType:
                yield SimpleNamespace(model_id=f"repo/{model_type.value}", model_type=model_type)

    async def _draw(self, _config, models, *, max_num_prompts=None, training_mode=None):
        model = await anext(models)
        task = self.add_task(model.model_type, training_mode)
        self.draws.append({"slot": (model.model_type, training_mode), "max_num_prompts": max_num_prompts})
        return task

    def add_task(self, model_type: ImageModelType, training_mode: ImageTrainingMode | None) -> SimpleNamespace:
        task = SimpleNamespace(
            task_id=f"task-{len(self.tasks) + 1}",
            task_type=TaskType.IMAGETASK,
            model_type=model_type,
            training_mode=training_mode,
            model_id=f"repo/{model_type.value}",
            model_params_count=0,
            status="pending",
        )
        self.tasks[task.task_id] = task
        return task

    def play_round_three(self, slot: tuple[ImageModelType, ImageTrainingMode]) -> None:
        """Round 3, already created, holding one task of the given slot."""
        self.rounds.append(
            TournamentRoundData(round_id=ROUND_THREE_ID, tournament_id=TID, round_number=3, round_type=RoundType.GROUP)
        )
        task = self.add_task(*slot)
        self.registered.append(
            TournamentTask(tournament_id=TID, round_id=ROUND_THREE_ID, task_id=task.task_id, group_id=f"{ROUND_THREE_ID}_group_001")
        )

    async def _register(self, tournament_tasks, _psql_db):
        self.registered.extend(tournament_tasks)

    async def _tournament_tasks(self, round_id, _psql_db):
        return [row for row in self.registered if row.round_id == round_id]

    async def _tournament_rounds(self, _tournament_id, _psql_db):
        return list(self.rounds)

    async def _get_task(self, task_id, _psql_db):
        return self.tasks.get(str(task_id))

    def slots(self) -> list[tuple[ImageModelType, ImageTrainingMode]]:
        return [draw["slot"] for draw in self.draws]

    def boss_round_slots(self) -> list[tuple[ImageModelType, ImageTrainingMode]]:
        return [
            (self.tasks[row.task_id].model_type, self.tasks[row.task_id].training_mode)
            for row in self.registered
            if row.round_id == BOSS_ROUND_ID
        ]


def _config():
    return MagicMock()


def _round(round_number: int) -> GroupRound:
    return GroupRound(
        round_id=f"tourn_round_{round_number:03d}", round_number=round_number, groups=[Group(member_ids=["a", "b", "c"])]
    )


def _boss_round() -> KnockoutRound:
    return KnockoutRound(round_id=BOSS_ROUND_ID, round_number=4, pairs=[("boss", "contender")])


def _assert_valid_boss_round(slots, round_three_slot) -> None:
    assert len(slots) == t_cst.IMAGE_BOSS_ROUND_TASKS == 4
    assert len(set(slots)) == 4
    assert set(slots) <= set(t_cst.FINAL_ROUND_IMAGE_TASK_SPECS)
    assert round_three_slot not in slots
    assert sum(1 for _, mode in slots if mode == EDIT) == 2
    assert sum(1 for _, mode in slots if mode == T2I) == 2


@pytest.mark.parametrize("round_number", [1, 2, 3])
async def test_planned_round_creates_one_task_drawn_from_its_slots(monkeypatch, round_number):
    world = World(monkeypatch, planned=True)
    round_data = _round(round_number)
    spec = t_cst.IMAGE_ROUND_PLAN[round_number - 1]

    created = await task_creator.create_image_tournament_tasks(round_data, TID, _config())

    assert len(created) == 1
    assert world.slots()[0] in spec.task_specs
    assert world.draws[0]["max_num_prompts"] == spec.max_num_prompts
    assert [(row.round_id, row.group_id, row.pair_id) for row in world.registered] == [
        (round_data.round_id, f"{round_data.round_id}_group_001", None)
    ]


async def test_round_one_is_a_capped_text_to_image_task(monkeypatch):
    world = World(monkeypatch, planned=True)

    await task_creator.create_image_tournament_tasks(_round(1), TID, _config())

    model_type, mode = world.slots()[0]
    assert model_type in t_cst.ROUND_ONE_IMAGE_MODEL_TYPES and mode == t_cst.ROUND_ONE_IMAGE_TRAINING_MODE == T2I
    assert world.draws[0]["max_num_prompts"] == t_cst.ROUND_ONE_IMAGE_MAX_SYNTH_PAIRS == 30


@pytest.mark.parametrize("slot", t_cst.FINAL_ROUND_IMAGE_TASK_SPECS)
async def test_round_three_can_draw_any_final_slot(monkeypatch, slot):
    world = World(monkeypatch, planned=True)
    monkeypatch.setattr(task_creator.random, "choice", lambda choices: slot if slot in choices else choices[0])

    await task_creator.create_image_tournament_tasks(_round(3), TID, _config())

    assert world.slots() == [slot]


async def test_planned_round_is_idempotent(monkeypatch):
    world = World(monkeypatch, planned=True)
    round_data = _round(2)

    first = await task_creator.create_image_tournament_tasks(round_data, TID, _config())
    second = await task_creator.create_image_tournament_tasks(round_data, TID, _config())

    assert second == first
    assert len(world.draws) == 1
    assert len(world.registered) == 1


async def test_a_round_outside_the_plan_is_refused(monkeypatch):
    World(monkeypatch, planned=True)

    with pytest.raises(ValueError, match="not a miner round"):
        await task_creator.create_image_tournament_tasks(_round(4), TID, _config())


def test_image_round_spec_matches_the_plan():
    assert [task_creator.image_round_spec(n) for n in (1, 2, 3)] == list(t_cst.IMAGE_ROUND_PLAN)
    with pytest.raises(ValueError):
        task_creator.image_round_spec(0)


@pytest.mark.parametrize("round_three_slot", t_cst.FINAL_ROUND_IMAGE_TASK_SPECS)
async def test_planned_boss_round_draws_two_edit_and_two_text_to_image_slots_around_round_three(
    monkeypatch, round_three_slot
):
    world = World(monkeypatch, planned=True)
    world.play_round_three(round_three_slot)

    created = await task_creator.create_image_tournament_tasks(_boss_round(), TID, _config(), is_final_round=True)

    assert len(created) == 4
    _assert_valid_boss_round(world.boss_round_slots(), round_three_slot)
    assert {draw["max_num_prompts"] for draw in world.draws} == {None}
    boss_rows = [row for row in world.registered if row.round_id == BOSS_ROUND_ID]
    assert {row.pair_id for row in boss_rows} == {f"{BOSS_ROUND_ID}_pair_001"}
    assert all(row.group_id is None for row in boss_rows)


async def test_planned_boss_round_with_round_three_on_edit_takes_both_other_edit_slots(monkeypatch):
    world = World(monkeypatch, planned=True)
    world.play_round_three((ImageModelType.HIDREAM_O1, EDIT))

    await task_creator.create_image_tournament_tasks(_boss_round(), TID, _config(), is_final_round=True)

    edit_slots = {slot for slot in world.boss_round_slots() if slot[1] == EDIT}
    assert edit_slots == {(ImageModelType.FLUX2_KLEIN_4B, EDIT), (ImageModelType.MOSS_IMAGE_BASE, EDIT)}


async def test_planned_boss_round_rerun_keeps_its_slots_and_creates_only_the_missing_ones(monkeypatch):
    world = World(monkeypatch, planned=True)
    round_three_slot = (ImageModelType.KREA2, T2I)
    world.play_round_three(round_three_slot)
    await task_creator.create_image_tournament_tasks(_boss_round(), TID, _config(), is_final_round=True)
    boss_rows = [row for row in world.registered if row.round_id == BOSS_ROUND_ID]
    kept = boss_rows[:2]
    kept_slots = {(world.tasks[row.task_id].model_type, world.tasks[row.task_id].training_mode) for row in kept}
    world.registered[:] = [row for row in world.registered if row.round_id != BOSS_ROUND_ID] + kept
    world.draws.clear()

    created = await task_creator.create_image_tournament_tasks(_boss_round(), TID, _config(), is_final_round=True)

    assert len(created) == 4
    assert len(world.draws) == 2
    assert kept_slots <= set(world.boss_round_slots())
    assert not kept_slots & set(world.slots())
    _assert_valid_boss_round(world.boss_round_slots(), round_three_slot)


async def test_planned_boss_round_creates_nothing_when_complete(monkeypatch):
    world = World(monkeypatch, planned=True)
    world.play_round_three((ImageModelType.MOSS_IMAGE_BASE, EDIT))
    await task_creator.create_image_tournament_tasks(_boss_round(), TID, _config(), is_final_round=True)
    world.draws.clear()

    await task_creator.create_image_tournament_tasks(_boss_round(), TID, _config(), is_final_round=True)

    assert world.draws == []


async def test_planned_boss_round_short_of_a_task_raises_and_the_next_pass_completes_it(monkeypatch):
    world = World(monkeypatch, planned=True)
    round_three_slot = (ImageModelType.FLUX2_KLEIN_4B, EDIT)
    world.play_round_three(round_three_slot)
    draw = world._draw
    failed_once: list[bool] = []

    async def first_slot_fails(config, models, **kwargs):
        if not failed_once:
            failed_once.append(True)
            raise ValueError("generation failed")
        return await draw(config, models, **kwargs)

    monkeypatch.setattr(task_creator, "create_synthetic_image_task", first_slot_fails)
    monkeypatch.setattr(task_creator.t_cst, "IMAGE_TASK_CREATION_MAX_ATTEMPTS", 1)
    with pytest.raises(RuntimeError, match="holds 3 of its 4 tasks"):
        await task_creator.create_image_tournament_tasks(_boss_round(), TID, _config(), is_final_round=True)
    assert len(world.boss_round_slots()) == 3

    created = await task_creator.create_image_tournament_tasks(_boss_round(), TID, _config(), is_final_round=True)

    assert len(created) == 4
    _assert_valid_boss_round(world.boss_round_slots(), round_three_slot)


async def test_planned_boss_round_without_a_round_three_task_is_refused(monkeypatch):
    world = World(monkeypatch, planned=True)

    with pytest.raises(ValueError, match="no round 3 task"):
        await task_creator.create_image_tournament_tasks(_boss_round(), TID, _config(), is_final_round=True)
    assert world.draws == []


async def test_three_task_round_one_structure_round_one_has_three_capped_text_to_image_tasks(monkeypatch):
    world = World(monkeypatch, planned=False)

    created = await task_creator.create_image_tournament_tasks(_round(1), TID, _config())

    assert len(created) == t_cst.SMALL_TOURNAMENT_GROUP_TASKS == 3
    assert world.slots() == [(model_type, T2I) for model_type in t_cst.ROUND_ONE_IMAGE_MODEL_TYPES]
    assert {draw["max_num_prompts"] for draw in world.draws} == {t_cst.ROUND_ONE_IMAGE_MAX_SYNTH_PAIRS}


async def test_three_task_round_one_structure_knockout_round_two_creates_one_task_per_pair(monkeypatch):
    world = World(monkeypatch, planned=False)

    created = await task_creator.create_image_tournament_tasks(
        KnockoutRound(round_id="tourn_round_002", round_number=2, pairs=[("a", "b")]), TID, _config()
    )

    assert len(created) == 1
    assert world.slots()[0] in t_cst.ROUND_TWO_IMAGE_TASK_SPECS
    assert [(row.group_id, row.pair_id) for row in world.registered] == [(None, "tourn_round_002_pair_001")]


async def test_three_task_round_one_structure_boss_round_has_the_six_final_slots(monkeypatch):
    world = World(monkeypatch, planned=False)

    created = await task_creator.create_image_tournament_tasks(_boss_round(), TID, _config(), is_final_round=True)

    assert len(created) == t_cst.FINAL_ROUND_IMAGE_TASKS == 6
    assert world.slots() == list(t_cst.FINAL_ROUND_IMAGE_TASK_SPECS)


def _replacement_world(monkeypatch, planned: bool, original_nodes: list, round_number: int):
    world = World(monkeypatch, planned=planned)
    original = world.add_task(ImageModelType.MOSS_IMAGE_BASE, EDIT)
    original.status = "prep_task_failure"
    world.rounds.append(
        TournamentRoundData(
            round_id=f"tourn_round_{round_number:03d}", tournament_id=TID, round_number=round_number, round_type=RoundType.GROUP
        )
    )
    monkeypatch.setattr(task_creator.seeds_sql, "is_seeded_text_tournament", AsyncMock(return_value=False))
    monkeypatch.setattr(task_creator.task_sql, "get_nodes_assigned_to_task", AsyncMock(return_value=original_nodes))
    monkeypatch.setattr(task_creator.task_sql, "get_expected_repo_name", AsyncMock(return_value="repo-name"))
    monkeypatch.setattr(task_creator.task_sql, "assign_node_to_task", AsyncMock())
    monkeypatch.setattr(task_creator.task_sql, "set_expected_repo_name", AsyncMock())
    monkeypatch.setattr(task_creator.task_sql, "detach_task_from_tournament", AsyncMock())
    return world, original


async def test_planned_round_replacement_keeps_the_slot_and_the_boss(monkeypatch):
    nodes = [SimpleNamespace(hotkey=EMISSION_BURN_HOTKEY), SimpleNamespace(hotkey="miner-1"), SimpleNamespace(hotkey="miner-2")]
    world, original = _replacement_world(monkeypatch, planned=True, original_nodes=nodes, round_number=3)

    new_task_id = await task_creator.replace_tournament_task(
        original.task_id, TID, ROUND_THREE_ID, f"{ROUND_THREE_ID}_group_001", None, _config()
    )

    assert world.slots() == [(ImageModelType.MOSS_IMAGE_BASE, EDIT)]
    assert world.draws[0]["max_num_prompts"] is None
    assigned = [(call.args[0], call.args[1].hotkey) for call in task_creator.task_sql.assign_node_to_task.await_args_list]
    assert assigned == [(new_task_id, node.hotkey) for node in nodes]
    assert [row.group_id for row in world.registered] == [f"{ROUND_THREE_ID}_group_001"]
    assert task_creator.task_sql.detach_task_from_tournament.await_args.args[:2] == (original.task_id, TID)


@pytest.mark.parametrize("planned", [True, False])
async def test_round_one_replacement_keeps_the_round_one_pair_cap(monkeypatch, planned):
    nodes = [SimpleNamespace(hotkey="miner-1")]
    world, original = _replacement_world(monkeypatch, planned=planned, original_nodes=nodes, round_number=1)
    original.model_type, original.training_mode = ImageModelType.Z_IMAGE, T2I

    await task_creator.replace_tournament_task(original.task_id, TID, "tourn_round_001", "tourn_round_001_group_001", None, _config())

    assert world.slots() == [(ImageModelType.Z_IMAGE, T2I)]
    assert world.draws[0]["max_num_prompts"] == t_cst.ROUND_ONE_IMAGE_MAX_SYNTH_PAIRS


async def test_boss_round_replacement_keeps_the_slot_with_the_default_size(monkeypatch):
    nodes = [SimpleNamespace(hotkey=EMISSION_BURN_HOTKEY), SimpleNamespace(hotkey="miner-1")]
    world, original = _replacement_world(monkeypatch, planned=True, original_nodes=nodes, round_number=4)
    world.rounds[-1] = world.rounds[-1].model_copy(update={"round_id": BOSS_ROUND_ID})

    await task_creator.replace_tournament_task(original.task_id, TID, BOSS_ROUND_ID, None, f"{BOSS_ROUND_ID}_pair_001", _config())

    assert world.slots() == [(ImageModelType.MOSS_IMAGE_BASE, EDIT)]
    assert world.draws[0]["max_num_prompts"] is None
    assert task_creator.task_sql.assign_node_to_task.await_count == 2


async def test_the_size_is_drawn_up_to_the_cap(monkeypatch):
    requested: list[int] = []

    async def generate(_config, num_prompts, _category):
        requested.append(num_prompts)
        return [SimpleNamespace(image_url=f"i{n}", text_url=f"t{n}") for n in range(num_prompts)], "prefix", None

    async def pool():
        yield SimpleNamespace(model_id="repo/z", model_type=ImageModelType.Z_IMAGE)

    monkeypatch.setattr(diffusion, "generate_image_synthetic_by_category", generate)
    monkeypatch.setattr(diffusion, "pick_image_synth_category", lambda: "style")
    monkeypatch.setattr(diffusion, "maybe_get_augmentation_config", lambda task_type: None)
    monkeypatch.setattr(diffusion, "add_task", AsyncMock(side_effect=lambda task, psql_db: task))
    monkeypatch.setattr(diffusion, "ImageRawTask", lambda **fields: SimpleNamespace(**fields))
    monkeypatch.setattr(diffusion, "TEMP_PATH_FOR_IMAGES", "/tmp/imgv2-b2-unused")
    monkeypatch.setattr(diffusion.Path, "mkdir", lambda *a, **k: None)

    await diffusion.create_synthetic_image_task(_config(), pool(), max_num_prompts=12, training_mode=T2I)

    assert synth_cst.MIN_IMAGE_SYNTH_PAIRS <= requested[0] <= 12
