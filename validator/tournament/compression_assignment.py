"""Which text round boss model each compression slot compresses: drawn once per text tournament, when its seeds are drawn.

The one random choice is which text round's model compression round 2 (the fixed round) compresses. The other two text
rounds feed the freeform rounds, the smaller-band model round 1 and the larger round 3, and the boss round's freeform and
fixed tasks follow from the choice (CompressionSourceAssignment). The draw is seeded by the text tournament id, so it is
reproducible, and recorded in compression_source_assignments, so a restart or a quant tournament bound later reads the same
one. A text tournament whose seed rows predate the table gets the earlier fixed mapping (text R2 fixed) recorded instead.

A text round whose model feeds any fixed-format slot draws its seed with the fixed rule (quant_shapes.fixed_compatible),
the others with the freeform rule.
"""

import random

from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.db.sql import quant_sources as quant_sources_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.tournament import constants as t_cst
from validator.tournament.models import TEXT_ROUND_ROLES_BY_SIZE
from validator.tournament.models import CompressionSourceAssignment
from validator.tournament.models import SeedRole


logger = get_logger(__name__)


def draw_compression_assignment(text_tournament_id: str) -> CompressionSourceAssignment:
    """The seeded draw for a text tournament: the same tournament id always gives the same assignment."""
    rng_seed = f"{text_tournament_id}:{t_cst.COMPRESSION_SOURCE_ASSIGNMENT_SEED_SALT}"
    fixed_text_role = random.Random(rng_seed).choice(TEXT_ROUND_ROLES_BY_SIZE)
    return CompressionSourceAssignment(text_tournament_id=text_tournament_id, fixed_text_role=fixed_text_role, rng_seed=rng_seed)


def legacy_compression_assignment(text_tournament_id: str) -> CompressionSourceAssignment:
    """The mapping of a text tournament seeded before assignments were drawn: text R1/R2/R3 -> compression R1/R2/R3."""
    return CompressionSourceAssignment(text_tournament_id=text_tournament_id, fixed_text_role=SeedRole.TEXT_R2)


async def compression_assignment(text_tournament_id: str, psql_db: PSQLDB) -> CompressionSourceAssignment:
    """The text tournament's recorded assignment; records one first when it has none (drawn, or the legacy mapping when the
    tournament already has seed rows) and returns what the database holds."""
    stored = await quant_sources_sql.get_compression_source_assignment(text_tournament_id, psql_db)
    if stored is not None:
        return stored
    if await seeds_sql.get_seed_task_history(text_tournament_id, psql_db):
        assignment = legacy_compression_assignment(text_tournament_id)
    else:
        assignment = draw_compression_assignment(text_tournament_id)
    stored = await quant_sources_sql.insert_compression_source_assignment(assignment, psql_db)
    small, large = stored.freeform_text_roles
    logger.info(
        f"Text tournament {text_tournament_id}: compression sources recorded - round 1 freeform {small.value}, round 2 fixed "
        f"{stored.fixed_text_role.value}, round 3 freeform {large.value} (rng_seed {stored.rng_seed!r})"
    )
    return stored
