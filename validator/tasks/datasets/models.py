"""Dataset payload models used by validator task preparation."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class DatasetData(BaseModel):
    dataset_id: str
    sparse_columns: list[str] = Field(default_factory=list)
    non_sparse_columns: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    author: str | None = None
    disabled: bool = False
    gated: bool = False
    last_modified: str | None = None
    likes: int = 0
    trending_score: int | None = None
    private: bool = False
    downloads: int = 0
    created_at: str | None = None
    description: str | None = None
    sha: str | None = None


class DatasetUrls(BaseModel):
    test_url: str
    train_url: str


class DatasetFiles(BaseModel):
    prefix: str
    data: str
    temp_path: Path | None = None


class DatasetJsons(BaseModel):
    train_data: list[Any]
    test_data: list[Any]

    def to_json_strings(self) -> dict[str, str]:
        return {
            "train_data": json.dumps(self.train_data),
            "test_data": json.dumps(self.test_data),
        }


class Dataset(BaseModel):
    dataset_id: str
    num_rows: int
    num_bytes_parquet_files: int
    dpo_available: bool = False
    dpo_prompt_column: str | None = None
    dpo_accepted_column: str | None = None
    dpo_rejected_column: str | None = None
