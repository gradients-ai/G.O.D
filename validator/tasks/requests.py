import asyncio
import json
import os
import random
import re
import tempfile
from typing import Callable

from huggingface_hub import HfApi
from pydantic import BaseModel

import validator.tasks.datasets.constants as cst
import validator.tasks.prep.constants as prep_cst
from core.constants.credentials import BUCKET_NAME
from core.downloads import download_s3_file
from core.interfaces import KeypairLike
from core.logging import get_logger
from core.logging import redact_url_queries
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import GrpoDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestQuant
from core.models.payload_models import TrainRequestText
from core.models.task_models import TaskStatus
from validator.evaluation.quant_loss import IGNORE_INDEX
from validator.evaluation.quant_loss import QuantEvalExample
from validator.evaluation.quant_loss import quant_eval_set_fingerprint
from validator.tasks.datasets.calibration import CalibrationUnavailableError
from validator.tasks.datasets.calibration import shared_split_seed
from validator.tasks.datasets.preparation import prepare_image_task
from validator.tasks.datasets.preparation import prepare_text_task
from validator.tasks.details import save_json_to_temp_file
from validator.tasks.details import upload_file_to_minio
from validator.tasks.models import AnyQuantRawTask
from validator.tasks.models import AnyTextTypeRawTask
from validator.tasks.models import ChatRawTask
from validator.tasks.models import DpoRawTask
from validator.tasks.models import GrpoRawTask
from validator.tasks.models import ImageRawTask
from validator.tasks.models import InstructTextRawTask
from validator.tasks.prep.yarn import prepare_yarn_extended_model
from validator.tasks.quant_eval_set import QuantEvalSetBuild
from validator.tasks.quant_eval_set import QuantTokenizer
from validator.tasks.quant_eval_set import build_quant_eval_set
from validator.tasks.quant_eval_set import quant_eval_max_sequence_len
from validator.tasks.quant_eval_set import quant_eval_set_jsonl


logger = get_logger(__name__)
hf_api = HfApi()


async def get_fake_text_dataset_size(task: AnyTextTypeRawTask) -> int:
    return 100_000


def _parse_model_size_from_name(model_id: str) -> int:
    """Best-effort size from a hub-style name like ``org/llama-70B-chat``.

    Requires a path separator or dash/underscore before the number so UUID fragments
    in continuous-SFT repo names (e.g. ``...c8068e356b7c...``) are not read as 356B.
    """
    match = re.search(r"(?:^|[-_/])(\d+(?:\.\d+)?)[bB](?:$|[-_/])", model_id)
    if not match:
        return 0
    return int(float(match.group(1)) * 1_000_000_000)


def get_model_num_params(model_id: str) -> int:
    try:
        model_info = hf_api.model_info(model_id)
        size = model_info.safetensors.total
        if size:
            return int(size)
    except Exception as e:
        logger.warning(f"Error getting model size from safetensors: {e}")

    model_size = _parse_model_size_from_name(model_id)
    if model_size:
        logger.info(f"Model size from name parse: {model_size}")
    else:
        logger.warning(f"Could not determine model size for {model_id}")
    return model_size


async def get_total_image_dataset_size(task: ImageRawTask) -> int:
    if task.image_text_pairs:
        return len(task.image_text_pairs)
    # Zip-backed image tasks are validated and counted during task prep.
    return 0


async def run_image_task_prep(task: ImageRawTask, keypair: KeypairLike | None = None, psql_db=None) -> ImageRawTask:
    test_url, train_url = await prepare_image_task(
        image_text_pairs=task.image_text_pairs,
        dataset_zip_url=task.ds if not task.image_text_pairs else None,
        training_mode=task.training_mode,
    )
    task.training_data = train_url
    task.test_data = test_url
    task.status = TaskStatus.LOOKING_FOR_NODES
    logger.info(
        "Data creation is complete - now time to find some miners",
    )
    return task


async def run_text_task_prep(
    task: AnyTextTypeRawTask,
    keypair: KeypairLike | None = None,
    psql_db=None,
) -> AnyTextTypeRawTask:
    if task.yarn_factor is not None:
        logger.info(f"Applying YaRN extension with factor {task.yarn_factor} to model {task.model_id}")
        try:
            yarn_model_id = await prepare_yarn_extended_model(task.model_id, task.yarn_factor)
            task.model_id = yarn_model_id
            logger.info(f"YaRN extension applied, new model_id: {yarn_model_id}")
        except Exception as e:
            logger.error(f"Failed to apply YaRN extension: {e}, using original model")
            # Clear yarn_factor to indicate extension was not applied
            task.yarn_factor = None

    split = await prepare_text_task(task, keypair=keypair, psql_db=psql_db)
    task.training_data = split.train_url
    task.status = TaskStatus.LOOKING_FOR_NODES
    task.synthetic_data = None  # TODO: remove from db schema
    task.test_data = split.test_url
    if split.calibration_url is not None:
        task.calibration_data = split.calibration_url

    if isinstance(task, InstructTextRawTask):
        task.field_instruction = cst.STANDARD_INSTRUCT_COLUMN
        task.field_output = cst.STANDARD_OUTPUT_COLUMN
        task.field_input = cst.STANDARD_INPUT_COLUMN if task.field_input else None
        task.field_system = cst.STANDARD_SYSTEM_COLUMN if task.field_system else None
    elif isinstance(task, DpoRawTask):
        task.field_prompt = cst.STANDARD_DPO_PROMPT_COLUMN
        task.field_chosen = cst.STANDARD_DPO_CHOSEN_COLUMN
        task.field_rejected = cst.STANDARD_DPO_REJECTED_COLUMN
        task.field_system = cst.STANDARD_SYSTEM_COLUMN if task.field_system else None
    elif isinstance(task, GrpoRawTask):
        task.field_prompt = cst.STANDARD_GRPO_PROMPT_COLUMN

    logger.info("Data creation is complete - now time to find some miners")
    return task


def _standard_instruct_dataset_type(dataset_type: InstructTextDatasetType) -> InstructTextDatasetType:
    """The dataset type naming the uploaded JSON's standard columns, as run_text_task_prep does for instruct."""
    return dataset_type.model_copy(
        update={
            "field_instruction": cst.STANDARD_INSTRUCT_COLUMN,
            "field_output": cst.STANDARD_OUTPUT_COLUMN,
            "field_input": cst.STANDARD_INPUT_COLUMN if dataset_type.field_input else None,
            "field_system": cst.STANDARD_SYSTEM_COLUMN if dataset_type.field_system else None,
        }
    )


async def _download_json_rows(url: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = await download_s3_file(url, save_path=temp_dir)
        with open(local_path) as f:
            return json.load(f)


def _eval_rows_seed(ds: str) -> str:
    """Seed of the eval-row cap: shared_split_seed, so the two tasks on one continuous chunk score identical rows."""
    return shared_split_seed(ds)


def _capped_eval_rows(rows: list[dict], seed: str) -> list[dict]:
    """At most MAX_TEST_DATA_POINTS rows: all of them at or under the cap, else a sample kept in original order.

    prepare_text_task's given-split path does not cap test rows (only its own split does), and a continuous
    chunk's test file is far larger. Seeded by ds, so the two tasks on one chunk score identical rows.
    """
    if len(rows) <= cst.MAX_TEST_DATA_POINTS:
        return rows
    kept = sorted(random.Random(seed).sample(range(len(rows)), cst.MAX_TEST_DATA_POINTS))
    return [rows[index] for index in kept]


def _default_tokenizer_loader() -> Callable[[str], QuantTokenizer]:
    # Deferred on purpose: quant_tokenizer imports transformers, an optional dependency (the `gpu` extra). The validator
    # image has it; the base install and the tests do not, so a top-level import would make this module unimportable there.
    from validator.tasks.quant_tokenizer import load_quant_tokenizer

    return load_quant_tokenizer


def _check_quant_eval_set(build: QuantEvalSetBuild) -> None:
    logger.info(
        f"Quant eval set: {len(build.examples)}/{build.n_rows} rows kept "
        f"(skipped: {build.n_skipped_empty} empty, {build.n_skipped_long} too long, {build.n_skipped_render} unrenderable)"
    )
    if len(build.examples) < prep_cst.QUANT_EVAL_MIN_EXAMPLES:
        raise ValueError(
            f"Quant eval set has {len(build.examples)} examples, minimum required is {prep_cst.QUANT_EVAL_MIN_EXAMPLES}"
        )


async def _upload_quant_jsonl(examples: list[QuantEvalExample], object_suffix: str, what: str) -> str:
    path, _ = await save_json_to_temp_file(quant_eval_set_jsonl(examples), prefix="quant_eval_", dump_json=False)
    try:
        url = await upload_file_to_minio(path, BUCKET_NAME, f"{os.urandom(8).hex()}_{object_suffix}")
    finally:
        os.remove(path)
    if not url:
        raise Exception(f"Failed to upload the {what} to MinIO storage")
    return url.strip('"')


async def _upload_quant_eval_set(build: QuantEvalSetBuild) -> str:
    return await _upload_quant_jsonl(build.examples, "quant_eval_set.jsonl", "quant eval set")


class QuantCalibrationPublication(BaseModel):
    """The tokenised calibration set a quant task (fixed or freeform) publishes: where it is, and what the request must carry."""

    url: str
    fingerprint: str
    n_examples: int


async def _build_and_upload_quant_calibration_set(
    raw_calibration_url: str,
    dataset_type: InstructTextDatasetType | ChatTemplateDatasetType,
    tokenizer: QuantTokenizer,
    max_sequence_len: int,
    eval_examples: list[QuantEvalExample],
) -> QuantCalibrationPublication:
    """One download, tokenise and upload of the calibration set; raises CalibrationUnavailableError below the minimum."""
    build = build_quant_eval_set(await _download_json_rows(raw_calibration_url), dataset_type, tokenizer, max_sequence_len)
    eval_keys = {(tuple(example.input_ids), tuple(example.labels)) for example in eval_examples}
    kept = [example for example in build.examples if (tuple(example.input_ids), tuple(example.labels)) not in eval_keys]
    if len(kept) < prep_cst.QUANT_CALIBRATION_MIN_EXAMPLES:
        raise CalibrationUnavailableError(
            f"Quant calibration set: only {len(kept)}/{build.n_rows} rows survive, "
            f"under {prep_cst.QUANT_CALIBRATION_MIN_EXAMPLES}"
        )
    fingerprint = quant_eval_set_fingerprint(kept)
    n_tokens = sum(sum(label != IGNORE_INDEX for label in example.labels[1:]) for example in kept)
    logger.info(
        f"Quant calibration set: {len(kept)}/{build.n_rows} rows kept, {n_tokens} supervised tokens, fingerprint {fingerprint}"
    )
    url = await _upload_quant_jsonl(kept, "quant_calibration_set.jsonl", "quant calibration set")
    return QuantCalibrationPublication(url=url, fingerprint=fingerprint, n_examples=len(kept))


async def _publish_quant_calibration_set(
    raw_calibration_url: str,
    dataset_type: InstructTextDatasetType | ChatTemplateDatasetType,
    tokenizer: QuantTokenizer,
    max_sequence_len: int,
    eval_examples: list[QuantEvalExample],
) -> QuantCalibrationPublication:
    """Tokenise the raw calibration rows with the eval set's own rules and upload them.

    An example equal to an eval example (token level) is dropped, so a calibration row is never a scored row. Fewer than
    QUANT_CALIBRATION_MIN_EXAMPLES survivors raise CalibrationUnavailableError at once (the outcome is deterministic). Any
    other failure is retried up to QUANT_CALIBRATION_PUBLISH_ATTEMPTS attempts in all, then raises CalibrationUnavailableError.
    The urls are presigned, so every message is redacted.
    """
    last_error = ""
    for attempt in range(1, prep_cst.QUANT_CALIBRATION_PUBLISH_ATTEMPTS + 1):
        try:
            return await _build_and_upload_quant_calibration_set(
                raw_calibration_url, dataset_type, tokenizer, max_sequence_len, eval_examples
            )
        except CalibrationUnavailableError as e:
            raise CalibrationUnavailableError(redact_url_queries(str(e))) from None
        except Exception as e:
            last_error = redact_url_queries(f"{type(e).__name__}: {e}")
            logger.warning(
                f"Quant calibration set: attempt {attempt}/{prep_cst.QUANT_CALIBRATION_PUBLISH_ATTEMPTS} failed: {last_error}"
            )
            if attempt < prep_cst.QUANT_CALIBRATION_PUBLISH_ATTEMPTS:
                await asyncio.sleep(prep_cst.QUANT_CALIBRATION_PUBLISH_BACKOFF_SECONDS * attempt)
    raise CalibrationUnavailableError(
        f"Quant calibration set could not be published after {prep_cst.QUANT_CALIBRATION_PUBLISH_ATTEMPTS} attempts: {last_error}"
    )


async def run_quant_task_prep(
    task: AnyQuantRawTask,
    keypair: KeypairLike | None = None,
    psql_db=None,
    tokenizer_loader: Callable[[str], QuantTokenizer] | None = None,
) -> AnyQuantRawTask:
    """Split and upload like a text task, then tokenise the test split into the scorer's JSONL and the calibration rows
    into the published calibration set.

    YaRN never applies (quant tasks have yarn_factor None). The tokenizer is the task's own base
    model's: augmentation never touches it and augmented_model_id is not set until model prep.
    The raw JSON test split is dropped: nothing reads it, and the rows are reproducible from ds + seed.
    Raises CalibrationUnavailableError when no calibration set can be published; the task is then left unchanged
    (every field is set only after the last step that can fail).
    """
    split = await prepare_text_task(task, keypair=keypair, psql_db=psql_db)
    if not split.calibration_url:
        raise CalibrationUnavailableError(f"Compression task {task.task_id} on {task.ds} has no calibration rows after data prep")
    rows = await _download_json_rows(split.test_url)
    tokenizer = (tokenizer_loader or _default_tokenizer_loader())(task.model_id)
    # The trainer receives task.dataset_type, so it must name the uploaded JSON's columns. It is set only
    # once prep succeeds: a failed task is persisted as is and recreated from its columns, which must
    # still name ds's raw columns for the re-split.
    dataset_type = task.dataset_type
    if isinstance(dataset_type, InstructTextDatasetType):
        dataset_type = _standard_instruct_dataset_type(dataset_type)

    max_sequence_len = quant_eval_max_sequence_len(tokenizer.context_len, prep_cst.QUANT_EVAL_MAX_SEQUENCE_LEN)
    build = build_quant_eval_set(_capped_eval_rows(rows, _eval_rows_seed(task.ds)), dataset_type, tokenizer, max_sequence_len)
    _check_quant_eval_set(build)

    calibration = await _publish_quant_calibration_set(
        split.calibration_url, dataset_type, tokenizer, max_sequence_len, build.examples
    )
    test_data = await _upload_quant_eval_set(build)

    task.dataset_type = dataset_type
    task.training_data = split.train_url
    task.test_data = test_data
    task.calibration_data = calibration.url
    task.calibration_fingerprint = calibration.fingerprint
    task.calibration_n_examples = calibration.n_examples
    task.synthetic_data = None
    task.status = TaskStatus.LOOKING_FOR_NODES
    logger.info("Quant data creation is complete - now time to find some miners")
    return task


def prepare_quant_task_request(task: AnyQuantRawTask) -> TrainRequestQuant:
    """Not available: compression train requests carry the calibration set and are built by the tournament orchestrator.

    The name stays because TaskConfig.task_request_prepare_function wires it for every quant task type; nothing calls it.
    """
    raise NotImplementedError(
        "compression train requests are built by the tournament orchestrator, which attaches the calibration set"
    )


def prepare_text_task_request(task: AnyTextTypeRawTask) -> TrainRequestText:
    if isinstance(task, InstructTextRawTask):
        dataset_type = InstructTextDatasetType(
            field_system=task.field_system,
            field_input=task.field_input,
            field_output=task.field_output,
            field_instruction=task.field_instruction,
            format=task.format,
            no_input_format=task.no_input_format,
        )
    elif isinstance(task, DpoRawTask):
        dataset_type = DpoDatasetType(
            field_prompt=task.field_prompt,
            field_system=task.field_system,
            field_chosen=task.field_chosen,
            field_rejected=task.field_rejected,
            prompt_format=task.prompt_format,
            chosen_format=task.chosen_format,
            rejected_format=task.rejected_format,
        )
    elif isinstance(task, GrpoRawTask):
        dataset_type = GrpoDatasetType(
            field_prompt=task.field_prompt,
            reward_functions=task.reward_functions,
            extra_column=task.extra_column,
        )
    elif isinstance(task, ChatRawTask):
        dataset_type = ChatTemplateDatasetType(
            chat_template=task.chat_template,
            chat_column=task.chat_column,
            chat_role_field=task.chat_role_field,
            chat_content_field=task.chat_content_field,
            chat_user_reference=task.chat_user_reference,
            chat_assistant_reference=task.chat_assistant_reference,
        )

    dataset = task.training_data if task.training_data else "dataset error"
    use_kl, kl_coef = (task.use_kl, task.kl_coef) if isinstance(task, InstructTextRawTask) else (False, None)
    task_request_body = TrainRequestText(
        dataset=dataset,
        model=task.model_id,
        dataset_type=dataset_type,
        file_format=FileFormat.S3,
        task_id=str(task.task_id),
        hours_to_complete=task.hours_to_complete,
        use_kl=use_kl,
        kl_coef=kl_coef,
    )

    return task_request_body


def prepare_image_task_request(task: ImageRawTask) -> TrainRequestImage:
    return TrainRequestImage(
        model=task.model_id,
        task_id=str(task.task_id),
        hours_to_complete=task.hours_to_complete,
        dataset_zip=task.training_data,
        model_type=task.model_type,
        training_mode=task.training_mode,
        trigger_word=task.trigger_word,
    )
