"""Schema for docs/incoming_changes.yaml, the miner-facing list of upcoming changes."""

import datetime
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PositiveInt
from pydantic import StringConstraints
from pydantic import field_validator
from pydantic import model_validator


REPO_ROOT = Path(__file__).resolve().parents[3]
INCOMING_CHANGES_PATH = REPO_ROOT / "docs" / "incoming_changes.yaml"
LABEL_PREFIX = "incoming:"

ChangeId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$", max_length=64)]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChangeArea(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    COMPRESSION = "compression"


class ChangeStatus(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    TESTING = "testing"
    SHIPPED = "shipped"
    POSTPONED = "postponed"


class IncomingChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: ChangeId = Field(description="Kebab-case id; the PR label `incoming:<id>` ships this entry.")
    title: NonEmptyText
    summary: NonEmptyText = Field(description="One or two miner-facing sentences.")
    why: NonEmptyText = Field(description="Why the change is being made.")
    areas: list[ChangeArea] = Field(min_length=1)
    status: ChangeStatus
    date: datetime.date = Field(description="The UTC Monday the change takes effect.")
    miner_action: NonEmptyText = Field(description='What miners need to do, or "Nothing".')
    prs: list[PositiveInt] = Field(default_factory=list, description="gradients-ai/G.O.D PR numbers.")
    docs: list[NonEmptyText] = Field(default_factory=list, description="Repo-relative doc paths.")
    shipped_on: datetime.date | None = None

    @field_validator("date", "shipped_on", mode="before")
    @classmethod
    def _iso_date_only(cls, value: object) -> object:
        # YAML parses bare YYYY-MM-DD into a date; anything else must be an ISO date string, not a timestamp.
        if isinstance(value, datetime.datetime):
            raise ValueError("must be an ISO date (YYYY-MM-DD), not a timestamp")
        if isinstance(value, str):
            return datetime.date.fromisoformat(value)
        return value

    @field_validator("date")
    @classmethod
    def _date_is_monday(cls, value: datetime.date) -> datetime.date:
        if value.weekday() != 0:
            raise ValueError(f"{value} is not a Monday")
        return value

    @field_validator("areas", "prs", "docs")
    @classmethod
    def _unique(cls, value: list) -> list:
        if len(set(value)) != len(value):
            raise ValueError("entries must be unique")
        return value

    @field_validator("docs")
    @classmethod
    def _repo_relative(cls, value: list[str]) -> list[str]:
        for path in value:
            if path.startswith("/") or ".." in Path(path).parts or "://" in path:
                raise ValueError(f"{path!r} must be a repo-relative path")
        return value

    @model_validator(mode="after")
    def _shipped_on_matches_status(self) -> "IncomingChange":
        if self.status == ChangeStatus.SHIPPED and self.shipped_on is None:
            raise ValueError("shipped entries need shipped_on")
        if self.status != ChangeStatus.SHIPPED and self.shipped_on is not None:
            raise ValueError("shipped_on must be null unless status is shipped")
        return self


class IncomingChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")

    changes: list[IncomingChange]

    @model_validator(mode="after")
    def _unique_ids(self) -> "IncomingChanges":
        counts = Counter(change.id for change in self.changes)
        duplicates = sorted(change_id for change_id, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate ids: {', '.join(duplicates)}")
        return self

    def get(self, change_id: str) -> IncomingChange | None:
        return next((change for change in self.changes if change.id == change_id), None)


def parse_incoming_changes(text: str) -> IncomingChanges:
    return IncomingChanges.model_validate(yaml.safe_load(text))


def load_incoming_changes(path: Path = INCOMING_CHANGES_PATH) -> IncomingChanges:
    return parse_incoming_changes(path.read_text())
