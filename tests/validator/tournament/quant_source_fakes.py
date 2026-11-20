"""In-memory stand-in for validator.db.sql.quant_sources, with the constraints the tables have: one text tournament feeds at most
one quant tournament, and a (quant tournament, slot) has at most one live (not superseded) task."""

from uuid import UUID

from validator.db.sql import quant_sources as quant_sources_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql.quant_sources import QuantTaskSourceInsert
from validator.tournament.models import QuantSourceSlot
from validator.tournament.models import QuantTaskSource
from validator.tournament.models import QuantTournamentSource


class FakeQuantSourceStore:
    def __init__(self) -> None:
        self.bindings: dict[str, QuantTournamentSource] = {}
        self.sources: list[QuantTaskSource] = []
        self.deleted: list[UUID] = []

    def install(self, monkeypatch) -> "FakeQuantSourceStore":
        for name in (
            "bind_quant_tournament",
            "get_quant_tournament_source",
            "get_bound_source_tournament_ids",
            "insert_quant_task_source",
            "get_live_quant_task_source",
            "get_quant_task_source",
            "list_quant_task_sources",
            "set_quant_task_source_model",
            "mark_quant_task_source_superseded",
            "repoint_quant_task_source",
        ):
            monkeypatch.setattr(quant_sources_sql, name, getattr(self, name))
        monkeypatch.setattr(task_sql, "delete_task", self.delete_task)
        return self

    async def delete_task(self, task_id, _db):
        """Deleting a task deletes its source row with it (ON DELETE CASCADE)."""
        self.deleted.append(UUID(str(task_id)))
        self.sources = [row for row in self.sources if row.quant_task_id != UUID(str(task_id))]

    def bind(self, quant_tournament_id: str, source_tournament_id: str) -> QuantTournamentSource:
        binding = QuantTournamentSource(quant_tournament_id=quant_tournament_id, source_tournament_id=source_tournament_id)
        self.bindings[quant_tournament_id] = binding
        return binding

    async def bind_quant_tournament(self, quant_tournament_id, source_tournament_id, _db):
        taken = {binding.source_tournament_id for binding in self.bindings.values()}
        if quant_tournament_id not in self.bindings and source_tournament_id not in taken:
            self.bind(quant_tournament_id, source_tournament_id)
        return self.bindings.get(quant_tournament_id)

    async def get_quant_tournament_source(self, quant_tournament_id, _db):
        return self.bindings.get(quant_tournament_id)

    async def get_bound_source_tournament_ids(self, _db):
        return {binding.source_tournament_id for binding in self.bindings.values()}

    def live(self, quant_tournament_id: str, slot: QuantSourceSlot) -> QuantTaskSource | None:
        return next(
            (
                row
                for row in self.sources
                if (row.quant_tournament_id, row.slot) == (quant_tournament_id, slot) and not row.superseded
            ),
            None,
        )

    async def insert_quant_task_source(self, source, _db, replacing_task_id=None):
        if replacing_task_id is not None:
            await self.mark_quant_task_source_superseded(replacing_task_id, _db)
        existing = self.live(source.quant_tournament_id, source.slot)
        if existing is not None and not source.superseded:
            return QuantTaskSourceInsert(source=existing, inserted=False)
        row = source.model_copy()
        self.sources.append(row)
        return QuantTaskSourceInsert(source=row, inserted=True)

    async def get_live_quant_task_source(self, quant_tournament_id, slot, _db):
        return self.live(quant_tournament_id, slot)

    async def get_quant_task_source(self, quant_task_id, _db):
        return next((row for row in self.sources if row.quant_task_id == UUID(str(quant_task_id))), None)

    async def list_quant_task_sources(self, quant_tournament_id, _db):
        return [row for row in self.sources if row.quant_tournament_id == quant_tournament_id]

    async def set_quant_task_source_model(self, quant_task_id, repo, is_fallback, _db):
        row = await self.get_quant_task_source(quant_task_id, _db)
        row.source_model_repo, row.source_is_fallback = repo, is_fallback

    async def mark_quant_task_source_superseded(self, quant_task_id, _db):
        row = await self.get_quant_task_source(quant_task_id, _db)
        if row is not None:
            row.superseded = True

    async def repoint_quant_task_source(self, quant_task_id, new_source_task_id, _db):
        row = await self.get_quant_task_source(quant_task_id, _db)
        if row is not None and row.source_model_repo is None:
            row.source_task_id = UUID(str(new_source_task_id))
