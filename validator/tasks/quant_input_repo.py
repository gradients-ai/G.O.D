"""Whether a Hugging Face repo holds a full model a quant task can take as its input.

A full model has a config.json, no LoRA adapter, and weights the evaluator can load. vLLM follows a
*.safetensors.index.json when one is present, so every shard the index maps must exist; a single complete
model.safetensors next to a stale index of missing shards loads in transformers but not in vLLM. Without an
index the repo needs at least one *.safetensors file.
"""

import asyncio
import json
import posixpath

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_files
from pydantic import BaseModel
from pydantic import ValidationError

from core.constants.credentials import HUGGINGFACE_TOKEN
from core.constants.quant import QUANT_ADAPTER_CONFIG_FILENAME
from core.constants.quant import QUANT_MODEL_CONFIG_FILENAME
from core.merged_repo import merged_repo_name


SAFETENSORS_SUFFIX = ".safetensors"
SAFETENSORS_INDEX_SUFFIX = ".safetensors.index.json"


class SafetensorsIndex(BaseModel):
    weight_map: dict[str, str]


def _missing_shards(index_name: str, index: SafetensorsIndex, files: set[str]) -> list[str]:
    folder = posixpath.dirname(index_name)
    shards = {posixpath.join(folder, shard) for shard in index.weight_map.values()}
    return sorted(shard for shard in shards if shard not in files)


async def _read_index(repo_id: str, index_name: str) -> SafetensorsIndex | str:
    """The parsed index, or why it cannot be parsed. Hub errors propagate."""
    path = await asyncio.to_thread(hf_hub_download, repo_id, index_name, token=HUGGINGFACE_TOKEN)
    try:
        with open(path) as f:
            return SafetensorsIndex.model_validate(json.load(f))
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as e:
        return f"{index_name} does not parse as a safetensors index: {e}"


async def is_adapter_repo(repo_id: str) -> bool:
    """Whether repo_id is a LoRA adapter (it has an adapter_config.json). Hub errors propagate like full_model_rejection's."""
    files = await asyncio.to_thread(list_repo_files, repo_id, token=HUGGINGFACE_TOKEN)
    return QUANT_ADAPTER_CONFIG_FILENAME in files


async def full_model_rejection(repo_id: str) -> str | None:
    """None when repo_id holds a loadable full model, else why it does not.

    Hub errors propagate (RepositoryNotFoundError for a missing repo), so each caller keeps its own policy
    for a missing repo versus a transient failure.
    """
    files = set(await asyncio.to_thread(list_repo_files, repo_id, token=HUGGINGFACE_TOKEN))
    if QUANT_ADAPTER_CONFIG_FILENAME in files:
        return f"has {QUANT_ADAPTER_CONFIG_FILENAME} (a LoRA adapter)"
    if QUANT_MODEL_CONFIG_FILENAME not in files:
        return f"has no {QUANT_MODEL_CONFIG_FILENAME}"

    index_names = sorted(name for name in files if name.endswith(SAFETENSORS_INDEX_SUFFIX))
    for index_name in index_names:
        index = await _read_index(repo_id, index_name)
        if isinstance(index, str):
            return index
        if not index.weight_map:
            return f"{index_name} maps no weights"
        missing = _missing_shards(index_name, index, files)
        if missing:
            return f"{index_name} references {len(missing)} shard(s) missing from the repo: {missing}"
    if not index_names and not any(name.endswith(SAFETENSORS_SUFFIX) for name in files):
        return f"has no {SAFETENSORS_SUFFIX} weights"
    return None


async def delete_merged_repo(merged_repo: str, raw_repo: str, base_model_id: str) -> None:
    """Delete a published adapter merge so the next merge job re-uploads it (the trainer skips the upload of a repo that
    already has a config and a weight file, so a half-uploaded one is never repaired otherwise).

    Refuses (ValueError) anything that is not exactly the repo the merge of raw_repo would publish, and never the raw adapter
    repo or the base model. Hub errors propagate.
    """
    owner = merged_repo.split("/", 1)[0]
    if merged_repo in (raw_repo, base_model_id) or merged_repo != merged_repo_name(raw_repo, owner):
        raise ValueError(f"refusing to delete {merged_repo}: it is not the merged repo of {raw_repo}")
    await asyncio.to_thread(HfApi(token=HUGGINGFACE_TOKEN).delete_repo, merged_repo, repo_type="model")
