"""The text -> compression source assignment: drawn once per text tournament, recorded, and followed by every compression slot.

Only which text round feeds compression round 2 (fixed) is random; round 1 takes the smaller of the other two text models and
round 3 the larger. The boss round's freeform task compresses the fixed round's model and its fixed task the round-3 model. A
text round whose model feeds any fixed slot draws its seed with the fixed shape rule, the others with the freeform rule.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg
import pytest

from core.models.task_models import TaskType
from tests.validator.tournament.quant_source_fakes import FakeQuantSourceStore
from tests.validator.tournament.test_compression_rounds import _Db
from tests.validator.tournament.test_compression_rounds import cluster  # noqa: F401  (fixture)
from tests.validator.tournament.test_quant_text_linkage import QUANT_TID
from tests.validator.tournament.test_quant_text_linkage import TEXT_TID
from tests.validator.tournament.test_quant_text_linkage import QuantWorld
from tests.validator.tournament.test_quant_text_linkage import _source
from tests.validator.tournament.test_seed_tasks import TID
from tests.validator.tournament.test_seed_tasks import env  # noqa: F401  (fixture)
from validator.db.sql import quant_sources as sources_sql
from validator.tasks import quant_shapes
from validator.tasks.models import RawTask
from validator.tournament import constants as t_cst
from validator.tournament import orchestrator
from validator.tournament import seed_tasks
from validator.tournament import task_creator
from validator.tournament.compression_assignment import compression_assignment
from validator.tournament.compression_assignment import draw_compression_assignment
from validator.tournament.compression_assignment import legacy_compression_assignment
from validator.tournament.models import ASSIGNED_QUANT_SLOTS
from validator.tournament.models import QUANT_SLOT_BY_SEED_ROLE
from validator.tournament.models import QUANT_SLOT_SPECS
from validator.tournament.models import TEXT_ROUND_ROLES_BY_SIZE
from validator.tournament.models import CompressionSourceAssignment
from validator.tournament.models import Group
from validator.tournament.models import GroupRound
from validator.tournament.models import QuantSourceSlot
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentSeedTask


R1, R2, R3 = SeedRole.TEXT_R1, SeedRole.TEXT_R2, SeedRole.TEXT_R3
MIGRATION = Path(__file__).parents[3] / "validator" / "db" / "migrations" / "20261115120000_compression_source_assignments.sql"
# fixed text role -> (compression R1, R2, R3, boss freeform, boss fixed) text roles
EXPECTED = {
    R1: (R2, R1, R3, R1, R3),
    R2: (R1, R2, R3, R2, R3),
    R3: (R1, R3, R2, R3, R2),
}
SLOT_ORDER = (
    QuantSourceSlot.R1,
    QuantSourceSlot.R2,
    QuantSourceSlot.R3,
    QuantSourceSlot.BOSS_R2_FREEFORM,
    QuantSourceSlot.BOSS_R3_FIXED,
)


def _assignment(fixed: SeedRole, text_tournament_id: str = TEXT_TID) -> CompressionSourceAssignment:
    return CompressionSourceAssignment(text_tournament_id=text_tournament_id, fixed_text_role=fixed, rng_seed="test")


# ------------------------------------------------------------------------------------------- the rule


class TestTheRule:
    @pytest.mark.parametrize("fixed", [R1, R2, R3])
    def test_only_the_fixed_round_source_is_chosen_and_the_freeform_rounds_take_the_smaller_model_first(self, fixed):
        assignment = _assignment(fixed)

        assert tuple(assignment.text_role(slot) for slot in SLOT_ORDER) == EXPECTED[fixed]
        small, large = assignment.freeform_text_roles
        assert TEXT_ROUND_ROLES_BY_SIZE.index(small) < TEXT_ROUND_ROLES_BY_SIZE.index(large)

    @pytest.mark.parametrize("fixed", [R1, R2, R3])
    def test_no_model_is_compressed_twice_on_the_same_track(self, fixed):
        assignment = _assignment(fixed)
        pairs = [(assignment.text_role(slot), QUANT_SLOT_SPECS[slot].task_type) for slot in ASSIGNED_QUANT_SLOTS]

        assert len(set(pairs)) == len(pairs)
        assert assignment.text_role(QuantSourceSlot.BOSS_R3_FIXED) != assignment.text_role(QuantSourceSlot.R2)

    @pytest.mark.parametrize("fixed", [R1, R2, R3])
    def test_the_fixed_round_model_and_the_round_three_model_are_fixed_fed(self, fixed):
        assignment = _assignment(fixed)
        small, large = assignment.freeform_text_roles

        assert assignment.track(fixed) == TaskType.QUANTFIXEDTASK
        assert assignment.track(large) == TaskType.QUANTFIXEDTASK  # the boss round's fixed task
        assert assignment.track(small) == TaskType.QUANTFREEFORMTASK

    def test_the_continuous_slots_keep_the_continuous_model(self):
        for fixed in (R1, R2, R3):
            assignment = _assignment(fixed)
            assert assignment.text_role(QuantSourceSlot.BOSS_CONTINUOUS_FREEFORM) == SeedRole.TEXT_CONTINUOUS
            assert assignment.text_role(QuantSourceSlot.BOSS_CONTINUOUS_FIXED) == SeedRole.TEXT_CONTINUOUS

    def test_the_track_order_of_the_rounds_is_unchanged(self):
        assert [spec.task_type for spec in t_cst.QUANT_ROUND_PLAN] == [
            TaskType.QUANTFREEFORMTASK,
            TaskType.QUANTFIXEDTASK,
            TaskType.QUANTFREEFORMTASK,
        ]
        assert t_cst.QUANT_ROUND_PLAN[1].quant_format == t_cst.QuantFormat.W2A16_ASYM_G64
        assert QUANT_SLOT_SPECS[QuantSourceSlot.BOSS_R2_FREEFORM].task_type == TaskType.QUANTFREEFORMTASK
        assert QUANT_SLOT_SPECS[QuantSourceSlot.BOSS_R3_FIXED].task_type == TaskType.QUANTFIXEDTASK

    def test_the_text_round_roles_are_ordered_by_their_size_band(self):
        plan = {spec.seed_role: spec.model_size_b for spec in t_cst.TEXT_ROUND_PLAN}
        bands = [plan[role] for role in TEXT_ROUND_ROLES_BY_SIZE]
        assert bands == sorted(bands)

    def test_only_a_text_round_role_can_be_the_fixed_source(self):
        with pytest.raises(ValueError):
            CompressionSourceAssignment(text_tournament_id="t", fixed_text_role=SeedRole.TEXT_CONTINUOUS)

    def test_the_legacy_mapping_is_text_rounds_onto_compression_rounds(self):
        assignment = legacy_compression_assignment("t")
        assert tuple(assignment.text_role(slot) for slot in SLOT_ORDER) == EXPECTED[R2]
        assert assignment.rng_seed is None


# ------------------------------------------------------------------------------------------- the draw and the record


class TestTheDraw:
    def test_the_draw_is_reproducible_from_the_tournament_id(self):
        first = draw_compression_assignment("text_tourn_a")

        assert first == draw_compression_assignment("text_tourn_a")
        assert first.rng_seed == f"text_tourn_a:{t_cst.COMPRESSION_SOURCE_ASSIGNMENT_SEED_SALT}"

    def test_every_text_round_can_be_drawn_for_the_fixed_round(self):
        drawn = {draw_compression_assignment(f"text_tourn_{i}").fixed_text_role for i in range(60)}
        assert drawn == set(TEXT_ROUND_ROLES_BY_SIZE)

    @pytest.mark.real_compression_assignment
    async def test_a_new_text_tournament_records_the_draw_and_later_reads_return_the_record(self, env, monkeypatch):  # noqa: F811
        store = FakeQuantSourceStore().install(monkeypatch)

        first = await compression_assignment(TID, None)

        assert first == draw_compression_assignment(TID) and store.assignments[TID] == first
        # A later read returns the record, whatever the draw would give now.
        monkeypatch.setattr(t_cst, "COMPRESSION_SOURCE_ASSIGNMENT_SEED_SALT", "another-salt")
        assert await compression_assignment(TID, None) == first

    @pytest.mark.real_compression_assignment
    async def test_a_text_tournament_seeded_before_the_table_records_the_legacy_mapping(self, env, monkeypatch):  # noqa: F811
        store = FakeQuantSourceStore().install(monkeypatch)
        env.store.rows.append(
            TournamentSeedTask(tournament_id=TID, role=R1, attempt=1, task_id=uuid4(), status=SeedStatus.TRAINING)
        )

        recorded = await compression_assignment(TID, None)

        assert recorded == legacy_compression_assignment(TID) and store.assignments[TID] == recorded


# ------------------------------------------------------------------------------------------- the text seed model rule


ODD_CONFIG = {  # dense and supported, but intermediate_size is not a multiple of 128
    "model_type": "qwen2",
    "hidden_size": 1536,
    "intermediate_size": 8964,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
}


@pytest.mark.real_compression_assignment
@pytest.mark.parametrize("fixed", [R1, R2, R3])
@pytest.mark.parametrize("role", [R1, R2, R3])
async def test_a_fixed_fed_text_round_rejects_a_non_divisible_model_and_a_freeform_fed_one_accepts_it(
    env,  # noqa: F811
    monkeypatch,
    tmp_path,
    fixed,
    role,
):
    store = FakeQuantSourceStore().install(monkeypatch)
    store.assignments[TID] = _assignment(fixed, TID)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(ODD_CONFIG))
    monkeypatch.setattr(quant_shapes, "hf_hub_download", lambda model_id, filename, token=None: str(config_path))
    monkeypatch.setattr(seed_tasks, "filter_quant_compatible", quant_shapes.filter_quant_compatible)

    async def one_model(_keypair, low, high):
        yield "org/odd"

    drawn: list[list[str]] = []

    async def make_instruct(config, models, datasets, **kwargs) -> RawTask:
        drawn.append([model_id async for model_id in models])
        return None

    monkeypatch.setattr(seed_tasks, "_get_text_models", one_model)
    monkeypatch.setattr(seed_tasks, "create_synthetic_instruct_text_task", make_instruct)

    await seed_tasks._draw_instruct(role, TID, env.config)

    fixed_fed = _assignment(fixed).track(role) == TaskType.QUANTFIXEDTASK
    assert drawn == [[] if fixed_fed else ["org/odd"]]


@pytest.mark.real_compression_assignment
async def test_a_failed_text_round_draws_its_own_task_with_the_same_rule(monkeypatch):
    store = FakeQuantSourceStore().install(monkeypatch)
    store.assignments["text_t"] = _assignment(R3, "text_t")
    tracks = []

    def capture(models, track):
        tracks.append(track)
        return models

    monkeypatch.setattr(task_creator, "filter_quant_compatible", capture)
    monkeypatch.setattr(task_creator, "_get_text_models", lambda _keypair, low, high: "pool")
    monkeypatch.setattr(task_creator, "_get_instruct_text_datasets", lambda _keypair: "datasets")
    monkeypatch.setattr(task_creator, "create_synthetic_instruct_text_task", AsyncMock())
    config = type("C", (), {"psql_db": None, "keypair": None})()

    for spec in t_cst.TEXT_ROUND_PLAN:
        await task_creator.draw_text_round_task(spec, "text_t", config)

    # fixed = text R3: compression R1 <- text R1 (freeform), R3 <- text R2 (freeform, and the boss fixed task)
    assert tracks == [TaskType.QUANTFREEFORMTASK, TaskType.QUANTFIXEDTASK, TaskType.QUANTFIXEDTASK]


# ------------------------------------------------------------------------------------------- the compression slots


@pytest.fixture
def assigned_world(env, monkeypatch):  # noqa: F811
    """A bound quant tournament whose text tournament recorded text R1 as the fixed round's source."""
    world = QuantWorld(env, monkeypatch)
    world.sources.assignments[TEXT_TID] = _assignment(R1)
    return world


@pytest.mark.real_compression_assignment
async def test_every_compression_slot_is_created_on_its_assigned_text_model(assigned_world):
    qw = assigned_world
    seeds = await qw.create_seeds()

    expected = dict(zip(SLOT_ORDER, EXPECTED[R1]))
    for role, seed in seeds.items():
        slot = QUANT_SLOT_BY_SEED_ROLE[role]
        text_role = expected.get(slot, SeedRole.TEXT_CONTINUOUS)
        task = qw.env.tasks[seed.task_id]
        assert task.upstream_task_id == qw.text_task(text_role).task_id, slot
        source = _source(qw, slot)
        assert (source.source_role, source.source_task_id) == (text_role, qw.text_task(text_role).task_id)
        assert task.task_type == QUANT_SLOT_SPECS[slot].task_type


@pytest.mark.real_compression_assignment
async def test_a_text_seed_replacement_relinks_the_assigned_slot(assigned_world):
    """Text R1 feeds compression round 2 (fixed) and the boss freeform task here: its replacement moves exactly those two."""
    qw = assigned_world
    task = await qw.create_round_task(2)
    old_text = qw.text_task(R1)
    assert task.upstream_task_id == old_text.task_id
    qw.training[str(old_text.task_id)] = "training"
    await orchestrator._release_chained_task(task, qw.config)

    old_text.status = "prep_task_failure"
    new_seed = await seed_tasks.replace_seed(qw.text_rows[R1], "prep", qw.config, prep_rejected=True)

    new_text = qw.env.tasks[new_seed.task_id]
    assert task.upstream_task_id == new_text.task_id
    assert _source(qw, QuantSourceSlot.R2).source_task_id == new_text.task_id
    assert _source(qw, QuantSourceSlot.BOSS_R2_FREEFORM).source_task_id == new_text.task_id
    for slot in (QuantSourceSlot.R1, QuantSourceSlot.R3, QuantSourceSlot.BOSS_R3_FIXED):
        assert _source(qw, slot).source_task_id != new_text.task_id


@pytest.mark.real_compression_assignment
async def test_a_failed_round_seed_plays_its_own_task_on_the_assigned_text_model(assigned_world):
    qw = assigned_world
    await qw.create_seeds()
    for row in qw.env.store.rows:
        if row.tournament_id == QUANT_TID and row.role == SeedRole.QUANT_R1:
            row.status = SeedStatus.FAILED
    round_data = GroupRound(round_id=f"{QUANT_TID}_round_001", round_number=1, groups=[Group(member_ids=["a", "b"])])

    (task,) = await task_creator._create_seed_input_round_task(round_data, QUANT_TID, qw.config)

    assert task.upstream_task_id == qw.text_task(R2).task_id  # compression R1 <- text R2 when text R1 is fixed
    assert _source(qw, QuantSourceSlot.R1).quant_task_id == task.task_id


# ------------------------------------------------------------------------------------------- the table


def test_the_migration_creates_the_table_and_drops_it_on_down():
    up, down = MIGRATION.read_text().split("-- migrate:down")
    assert "CREATE TABLE IF NOT EXISTS compression_source_assignments" in up
    assert "CHECK (fixed_text_role IN ('text_r1', 'text_r2', 'text_r3'))" in up
    assert "DROP TABLE IF EXISTS compression_source_assignments;" in down


async def test_the_first_recorded_assignment_wins_and_reads_back(cluster):  # noqa: F811
    connection = await cluster.connect()
    try:
        await connection.execute("DROP TABLE IF EXISTS compression_source_assignments, tournaments")
        await connection.execute("CREATE TABLE tournaments (tournament_id text PRIMARY KEY)")
        await connection.execute("INSERT INTO tournaments VALUES ('text_a')")
        await connection.execute(MIGRATION.read_text().split("-- migrate:down")[0])
        db = _Db(connection)

        assert await sources_sql.get_compression_source_assignment("text_a", db) is None
        first = await sources_sql.insert_compression_source_assignment(draw_compression_assignment("text_a"), db)
        other = _assignment(R3 if first.fixed_text_role != R3 else R1, "text_a")
        again = await sources_sql.insert_compression_source_assignment(other, db)

        assert again.fixed_text_role == first.fixed_text_role == draw_compression_assignment("text_a").fixed_text_role
        assert first.rng_seed == draw_compression_assignment("text_a").rng_seed and first.created_at is not None
        assert await sources_sql.get_compression_source_assignment("text_a", db) == first
        with pytest.raises(asyncpg.CheckViolationError):
            await connection.execute(
                "INSERT INTO compression_source_assignments (text_tournament_id, fixed_text_role) "
                "VALUES ('text_a2', 'text_continuous')"
            )
    finally:
        await connection.close()
