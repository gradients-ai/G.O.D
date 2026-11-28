import asyncio
import json
import math
import random
import re
import tempfile
import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import AsyncGenerator

import httpx

import validator.tasks.synthetics.concepts as concepts
import validator.tasks.synthetics.constants as synth_cst
from core.logging import get_logger
from core.models.dataset_models import ImageTextPair
from core.models.image_models import IMAGE_MODEL_CAPABILITIES
from core.models.image_models import ImageModelType
from core.models.image_models import ImageTrainingMode
from core.models.payload_models import ImageModelInfo
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.app.config import Config
from validator.db.sql.tasks import add_task
from validator.infrastructure.fal_client import download_url
from validator.infrastructure.fal_client import extract_image_urls
from validator.infrastructure.fal_client import extract_text
from validator.infrastructure.fal_client import persist_image_edit_pairs
from validator.infrastructure.fal_client import persist_image_text_pairs
from validator.infrastructure.fal_client import post_to_fal
from validator.infrastructure.fal_client import upload_local_file
from validator.infrastructure.service_constants import NULL_ACCOUNT_ID
from validator.tasks.datasets.constants import TEMP_PATH_FOR_IMAGES
from validator.tasks.details import retry_with_backoff
from validator.tasks.models import ImageRawTask
from validator.tasks.models import RawTask
from validator.tasks.prep.augmentation import maybe_get_augmentation_config


logger = get_logger(__name__)




# Person held-out prompts come back under this JSON key and are generated after the training prompts, so the task's
# last IMAGE_SYNTH_TEST_PAIRS pairs are its test split (see train_test_split_image). Other kinds: concepts.py.
HELD_OUT_KEY = "test_prompts"


@dataclass(frozen=True)
class TriggeredPromptSet:
    trigger: str
    prompts: list[str]
    test_prompts: list[str] = field(default_factory=list)


def _load_json_from_text(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            raise ValueError(f"Failed to extract JSON from model response: {text}")
        return json.loads(json_match.group(0))


def _get_prompts_from_response(text: str, num_prompts: int) -> list[str]:
    result = _load_json_from_text(text)
    prompts = result.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError(f"Prompt response missing prompts list: {text}")

    clean_prompts = [prompt.strip() for prompt in prompts if isinstance(prompt, str) and prompt.strip()]
    if len(clean_prompts) < num_prompts:
        raise ValueError(f"Generated {len(clean_prompts)} prompts, expected at least {num_prompts}")
    return clean_prompts[:num_prompts]


def _normalize_trigger_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _clean_trigger(value: object, response_text: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Prompt response missing non-empty trigger/name: {response_text}")
    return re.sub(r"\s+", " ", value).strip()


def _get_clean_prompt_list(result: dict, prompt_key: str, num_prompts: int, response_text: str) -> list[str]:
    prompts = result.get(prompt_key)
    if not isinstance(prompts, list):
        raise ValueError(f"Prompt response missing {prompt_key} list: {response_text}")

    clean_prompts = [prompt.strip() for prompt in prompts if isinstance(prompt, str) and prompt.strip()]
    if len(clean_prompts) < num_prompts:
        raise ValueError(f"Generated {len(clean_prompts)} prompts, expected at least {num_prompts}")
    return clean_prompts[:num_prompts]


def _validate_trigger_in_prompts(trigger: str, prompts: list[str], context: str) -> None:
    normalized_trigger = _normalize_trigger_text(trigger)
    missing_indexes = [
        index + 1 for index, prompt in enumerate(prompts) if normalized_trigger not in _normalize_trigger_text(prompt)
    ]
    if missing_indexes:
        raise ValueError(f"{context} prompts missing trigger '{trigger}' at indexes: {missing_indexes}")


def _get_triggered_prompt_set_from_response(
    text: str, num_prompts: int, prompt_key: str = "prompts", context: str = "image", num_test: int = 0
) -> TriggeredPromptSet:
    result = _load_json_from_text(text)
    trigger = _clean_trigger(result.get("trigger") or result.get("name"), text)
    prompts = _get_clean_prompt_list(result, prompt_key, num_prompts, text)
    _validate_trigger_in_prompts(trigger, prompts, context)
    test_prompts = _get_clean_prompt_list(result, HELD_OUT_KEY, num_test, text) if num_test else []
    _validate_trigger_in_prompts(trigger, test_prompts, f"{context} held-out")
    return TriggeredPromptSet(trigger=trigger, prompts=prompts, test_prompts=test_prompts)


def _sanitize_ds_fragment(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _concept_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z]{4,}", left.casefold()))
    right_tokens = set(re.findall(r"[a-z]{4,}", right.casefold()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


async def _post_to_fal_text(prompt: str) -> str:
    result = await post_to_fal(
        synth_cst.FAL_TEXT_PROMPT_MODEL,
        {
            "prompt": prompt,
            "model": synth_cst.FAL_TEXT_PROMPT_LLM,
            "temperature": synth_cst.FAL_TEXT_PROMPT_TEMPERATURE,
        },
    )
    return extract_text(result)


async def _get_face_reference_url() -> str:
    Path(TEMP_PATH_FOR_IMAGES).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TEMP_PATH_FOR_IMAGES) as tmp_dir:
        reference_path = await download_url(synth_cst.IMAGE_SYNTH_FACE_IMAGE_URL, Path(tmp_dir) / "face")
        return await upload_local_file(reference_path, "image_synth/face_refs")


def _person_prompt_request(num_prompts: int, num_test: int = 0) -> str:
    held_out = concepts.CONCEPT_TEMPLATES["person_held_out"].format(num_test=num_test) if num_test else ""
    return _person_prompt_body(num_prompts) + held_out


def _person_prompt_body(num_prompts: int) -> str:
    return f"""Generate {num_prompts} different prompts for realistic avatar images of the person in the provided image.

Requirements:
- Invent one natural fictional name for the person. Every prompt must include that exact same name.
- Do not mention age or age category in any prompt.
- Do not use words like child, teen, adult, elderly, young, old, or any numeric age.
- Preserve visible identity cues like face shape, hair, skin tone, and gender presentation.
- Place the person in varied safe public or professional settings, backgrounds, wardrobes, lighting, and camera compositions.
- Use safe neutral or positive expressions like smiling, thoughtful, calm, confident, surprised, relaxed, focused, or joyful.
- Avoid threatening, fearful, sexualized, violent, vulnerable, or unsafe scenarios.
- Avoid bedrooms, schools, bathrooms, dim hallways, alleys, distress, anger, fear, injury, weapons, or intimidation.
- Each prompt should read like a high-quality text-to-image prompt: concise but descriptive, with subject, setting,
  clothing, action or emotion, lighting, composition, camera/lens feel, and a realistic photographic style.
- Avoid repeating the same scene, outfit, emotion, or composition across prompts.

Return JSON only:
{{"name": "fictional name", "prompts": ["prompt 1", "prompt 2"]}}"""


@retry_with_backoff
async def generate_person_prompts_with_fal_vision(
    face_image_url: str, num_prompts: int, num_test: int = 0
) -> TriggeredPromptSet:
    logger.info(
        f"Generating {num_prompts} person prompts with "
        f"{synth_cst.FAL_PERSON_PROMPT_MODEL}/{synth_cst.FAL_PERSON_PROMPT_VLM}"
    )
    payload = {
        "image_urls": [face_image_url],
        "prompt": _person_prompt_request(num_prompts, num_test),
        "model": synth_cst.FAL_PERSON_PROMPT_VLM,
    }
    result = await post_to_fal(synth_cst.FAL_PERSON_PROMPT_MODEL, payload)
    prompt_set = _get_triggered_prompt_set_from_response(
        extract_text(result), num_prompts, context="person", num_test=num_test
    )
    logger.info(f"Generated {len(prompt_set.prompts)} person prompts with trigger: {prompt_set.trigger}")
    return prompt_set


def _fal_image_payload(model_id: str, prompt: str, reference_image_url: str | None = None) -> dict:
    if model_id == synth_cst.FAL_AVATAR_MODEL:
        if not reference_image_url:
            raise ValueError("reference_image_url is required for avatar generation")
        return {
            "prompt": prompt,
            "image_urls": [reference_image_url],
            "num_images": 1,
            "resolution": synth_cst.FAL_NANO_BANANA_RESOLUTION,
            "output_format": synth_cst.FAL_IMAGE_OUTPUT_FORMAT,
        }

    if model_id == synth_cst.FAL_STYLE_MODEL_GPT_IMAGE_2:
        return {
            "prompt": prompt,
            "quality": synth_cst.FAL_GPT_IMAGE_2_QUALITY,
            "num_images": 1,
            "output_format": synth_cst.FAL_IMAGE_OUTPUT_FORMAT,
        }

    return {
        "prompt": prompt,
        "num_images": 1,
        "resolution": synth_cst.FAL_NANO_BANANA_RESOLUTION,
        "output_format": synth_cst.FAL_IMAGE_OUTPUT_FORMAT,
    }


@retry_with_backoff
async def generate_fal_image(model_id: str, prompt: str, reference_image_url: str | None = None) -> str:
    result = await post_to_fal(model_id, _fal_image_payload(model_id, prompt, reference_image_url))
    return extract_image_urls(result)[0]


async def generate_fal_images_for_prompts(
    model_id: str, prompts: list[str], reference_image_url: str | None = None
) -> list[tuple[str, str]]:
    semaphore = asyncio.Semaphore(synth_cst.FAL_IMAGE_GENERATION_CONCURRENCY)

    async def generate_one(index: int, prompt: str) -> tuple[str, str]:
        async with semaphore:
            logger.info(f"Generating image {index + 1}/{len(prompts)} with {model_id}")
            image_url = await generate_fal_image(model_id, prompt, reference_image_url)
            logger.info(f"Generated image {index + 1}/{len(prompts)} with {model_id}")
            return image_url, prompt

    logger.info(
        f"Generating {len(prompts)} images with {model_id}, concurrency={synth_cst.FAL_IMAGE_GENERATION_CONCURRENCY}"
    )
    return await asyncio.gather(*(generate_one(index, prompt) for index, prompt in enumerate(prompts)))


def _triggered_ds_prefix(prefix: str, trigger: str) -> str:
    trigger_fragment = _sanitize_ds_fragment(trigger)
    return f"{prefix}_{trigger_fragment}" if trigger_fragment else prefix


def _to_ideogram_aspect_ratio(width: int = 1024, height: int = 1024) -> str:
    divisor = math.gcd(width, height) or 1
    return f"{width // divisor}x{height // divisor}"


def _ordered_dict(data: dict, order: tuple[str, ...]) -> dict:
    known = [key for key in order if key in data]
    extra = [key for key in data if key not in order]
    return {key: data[key] for key in (*known, *extra)}


def _canonical_ideogram_caption(caption: dict) -> dict:
    """Match Ideogram's JSON prompt key order and remove fields not used for training captions."""
    caption.pop("aspect_ratio", None)
    caption = _ordered_dict(
        caption,
        ("high_level_description", "style_description", "compositional_deconstruction"),
    )

    style_description = caption.get("style_description")
    if isinstance(style_description, dict):
        caption["style_description"] = _ordered_dict(
            style_description,
            ("aesthetics", "lighting", "photo", "art_style", "medium", "color_palette"),
        )

    composition = caption.get("compositional_deconstruction")
    if isinstance(composition, dict):
        composition = _ordered_dict(composition, ("background", "elements"))
        elements = composition.get("elements")
        if isinstance(elements, list):
            normalized_elements = []
            for element in elements:
                if isinstance(element, dict):
                    element.pop("bbox", None)
                    element = _ordered_dict(element, ("type", "desc"))
                normalized_elements.append(element)
            composition["elements"] = normalized_elements
        caption["compositional_deconstruction"] = composition

    return caption


@retry_with_backoff
async def generate_ideogram_json_prompt(prompt: str, api_key: str | None = None) -> str:
    api_key = api_key or synth_cst.IDEOGRAM_MAGIC_PROMPT_API_KEY
    if not api_key:
        raise RuntimeError("IDEOGRAM_API_KEY or MAGIC_PROMPT_API_KEY must be set for ideogram4 image tasks")

    async with httpx.AsyncClient(timeout=synth_cst.IDEOGRAM_MAGIC_PROMPT_TIMEOUT_SECONDS) as client:
        response = await client.post(
            synth_cst.IDEOGRAM_MAGIC_PROMPT_URL,
            headers={"Api-Key": api_key, "Content-Type": "application/json"},
            json={
                "text_prompt": prompt,
                "aspect_ratio": _to_ideogram_aspect_ratio(synth_cst.MIN_IMAGE_WIDTH, synth_cst.MIN_IMAGE_HEIGHT),
            },
        )
        response.raise_for_status()

    data = response.json()
    json_prompt = data.get("json_prompt")
    if not isinstance(json_prompt, dict):
        raise RuntimeError(f"Ideogram magic prompt returned no json_prompt: {data}")

    return json.dumps(_canonical_ideogram_caption(json_prompt), ensure_ascii=False, separators=(",", ":"))


async def rewrite_pairs_with_ideogram_json_prompts(image_text_pairs: list[ImageTextPair]) -> list[ImageTextPair]:
    semaphore = asyncio.Semaphore(synth_cst.IDEOGRAM_MAGIC_PROMPT_CONCURRENCY)
    Path(TEMP_PATH_FOR_IMAGES).mkdir(parents=True, exist_ok=True)

    async def rewrite_one(index: int, pair: ImageTextPair, work_dir: Path) -> ImageTextPair:
        async with semaphore:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.get(pair.text_url)
                response.raise_for_status()
            json_prompt = await generate_ideogram_json_prompt(response.text.strip())
            text_path = work_dir / f"{index}.txt"
            text_path.write_text(json_prompt)
            text_url = await upload_local_file(text_path, "image_synth/ideogram_prompts")
            return ImageTextPair(image_url=pair.image_url, text_url=text_url)

    logger.info(
        f"Rewriting {len(image_text_pairs)} prompts with Ideogram magic prompt, "
        f"concurrency={synth_cst.IDEOGRAM_MAGIC_PROMPT_CONCURRENCY}"
    )
    with tempfile.TemporaryDirectory(dir=TEMP_PATH_FOR_IMAGES) as tmp_dir:
        work_dir = Path(tmp_dir)
        tasks = [rewrite_one(index, pair, work_dir) for index, pair in enumerate(image_text_pairs)]
        return await asyncio.gather(*tasks)


async def generate_edit_synthetic(num_prompts: int, num_test: int = 0) -> tuple[list[ImageTextPair], str, None]:
    """One invented edit rule: unbranded source photos, targets edited with the full rule, and a caption that only
    names the rule. Held-out pairs use object types that never appear in training and come last."""
    spec = await generate_rule_edit_spec(num_prompts, num_test)
    subjects = spec.train_subjects + spec.test_subjects

    async def make_pair(subject: str) -> tuple[str, str, str]:
        source_url = await generate_fal_image(synth_cst.FAL_STYLE_MODEL_GPT_IMAGE_2, concepts.rule_edit_source_prompt(subject))
        target_url = await generate_fal_image(synth_cst.FAL_AVATAR_MODEL, spec.rule_spec, source_url)
        return source_url, target_url, spec.instruction

    logger.info(f"Generating {len(subjects)} pairs for edit rule '{spec.goal_name}' ({num_test} held out)")
    generated_pairs = await concepts.gather_limited(make_pair(subject) for subject in subjects)
    image_text_pairs = await persist_image_edit_pairs(generated_pairs)
    logger.info("Persisted %s pairs for edit rule '%s': %s", len(image_text_pairs), spec.goal_name, spec.rule_spec)
    return image_text_pairs, _triggered_ds_prefix(synth_cst.EDIT_SYNTH_DS_PREFIX, spec.goal_name), None


async def generate_person_synthetic(num_prompts: int, num_test: int = 0) -> tuple[list[ImageTextPair], str, str]:
    logger.info("Fetching and uploading person reference image")
    face_image_url = await _get_face_reference_url()
    prompt_set = await generate_person_prompts_with_fal_vision(face_image_url, num_prompts, num_test)
    image_prompt_pairs = await generate_fal_images_for_prompts(
        synth_cst.FAL_AVATAR_MODEL, prompt_set.prompts + prompt_set.test_prompts, face_image_url
    )
    logger.info(f"Persisting {len(image_prompt_pairs)} person image-text pairs")
    image_text_pairs = await persist_image_text_pairs(image_prompt_pairs)
    logger.info(f"Persisted {len(image_text_pairs)} person image-text pairs")
    return image_text_pairs, _triggered_ds_prefix(synth_cst.PERSON_SYNTH_DS_PREFIX, prompt_set.trigger), prompt_set.trigger


@retry_with_backoff
async def generate_concept_spec(kind: str, num_train: int, num_test: int) -> concepts.ConceptSpec:
    brief = concepts.creative_brief(kind)
    request = concepts.concept_request(kind, num_train, num_test, brief, list(concepts.RECENT_CONCEPTS[kind]))
    logger.info(f"Calling FAL text prompt model for a {kind} concept ({brief.get('brief', '')})")
    spec = concepts.parse_concept(kind, _load_json_from_text(await _post_to_fal_text(request)), num_train, num_test)
    if concepts.similar_to_recent(kind, f"{spec.trigger} {spec.identity_spec}", _concept_similarity):
        raise ValueError(f"Generated {kind} concept is too similar to a recent task")
    concepts.RECENT_CONCEPTS[kind].append(f"{spec.trigger} {spec.identity_spec}")
    return spec


async def _render_concept(spec: concepts.ConceptSpec) -> list[tuple[str, str]]:
    """Reference image first, then every training and held-out image rendered from it; returns (url, caption)."""
    reference_url = await _verified_reference(spec) if spec.reference_prompt else None
    jobs = concepts.render_jobs(spec, reference_url)
    logger.info(f"Rendering {len(jobs)} {spec.kind} images for {spec.trigger} ({len(spec.test_contexts)} held out)")
    urls = await concepts.gather_limited(_render_checked(spec, job) for job in jobs)
    return [(url, caption) for url, (_, _, _, caption) in zip(urls, jobs)]


async def _ask_vision(image_urls: list[str], prompt: str) -> dict:
    payload = {"image_urls": image_urls, "prompt": prompt, "model": synth_cst.FAL_PERSON_PROMPT_VLM}
    return _load_json_from_text(extract_text(await post_to_fal(synth_cst.FAL_PERSON_PROMPT_MODEL, payload)))


async def _verified_reference(spec: concepts.ConceptSpec) -> str:
    """Generate the reference and confirm it shows the described concept before anything is rendered from it."""
    for attempt in range(synth_cst.IMAGE_SYNTH_QA_RETRIES + 1):
        logger.info(f"Generating {spec.kind} reference image for {spec.trigger} (attempt {attempt + 1})")
        url = await generate_fal_image(synth_cst.FAL_STYLE_MODEL_GPT_IMAGE_2, concepts.reference_prompt(spec))
        verdict = await _ask_vision([url], concepts.reference_qa_request(spec))
        if verdict.get("matches") is True:
            return url
        logger.info(f"Reference for {spec.trigger} does not match its spec: {verdict.get('reason', '')}")
    raise ValueError(f"{spec.kind} reference for {spec.trigger} never matched its spec")


async def _check_against_reference(spec: concepts.ConceptSpec, reference_url: str, image_url: str, caption: str) -> dict:
    return await _ask_vision([reference_url, image_url], concepts.qa_request(spec, caption))


async def _render_checked(spec: concepts.ConceptSpec, job: tuple[str, str | None, str, str]) -> str:
    """Render one image; when it is rendered from the reference, re-render until a vision check confirms it still
    shows the same concept, clearly visible. Raises after IMAGE_SYNTH_QA_RETRIES re-renders, rejecting the concept."""
    model, reference_url, prompt, caption = job
    for attempt in range(synth_cst.IMAGE_SYNTH_QA_RETRIES + 1):
        url = await generate_fal_image(model, prompt, reference_url)
        if reference_url is None:
            return url
        verdict = await _check_against_reference(spec, reference_url, url, caption)
        if concepts.qa_passed(verdict):
            return url
        logger.info(f"Consistency check failed for '{caption}' (attempt {attempt + 1}): {verdict.get('reason', '')}")
    raise ValueError(f"{spec.kind} image '{caption}' failed the consistency check {synth_cst.IMAGE_SYNTH_QA_RETRIES + 1} times")


async def generate_concept_synthetic(
    kind: str, num_prompts: int, num_test: int = 0
) -> tuple[list[ImageTextPair], str, str]:
    """num_prompts training pairs then num_test held-out pairs for one locked concept (see concepts.py).
    A concept whose images keep failing the consistency check is dropped and a new concept is generated."""
    for attempt in range(synth_cst.IMAGE_SYNTH_CONCEPT_ATTEMPTS):
        spec = await generate_concept_spec(kind, num_prompts, num_test)
        try:
            image_caption_pairs = await _render_concept(spec)
            break
        except ValueError as error:
            logger.warning(f"Dropping {kind} concept {spec.trigger} (attempt {attempt + 1}): {error}")
    else:
        raise ValueError(f"No {kind} concept passed the consistency check in {synth_cst.IMAGE_SYNTH_CONCEPT_ATTEMPTS} attempts")
    image_text_pairs = await persist_image_text_pairs(image_caption_pairs)
    logger.info(f"Persisted {len(image_text_pairs)} {kind} pairs for {spec.trigger}")
    return image_text_pairs, _triggered_ds_prefix(f"{kind}", spec.trigger), spec.trigger


@retry_with_backoff
async def generate_rule_edit_spec(num_train: int, num_test: int) -> concepts.RuleEditSpec:
    brief = concepts.creative_brief("edit")
    request = concepts.concept_request("edit", num_train, num_test, brief, list(concepts.RECENT_CONCEPTS["edit"]))
    logger.info(f"Calling FAL text prompt model for an edit rule ({brief['brief']})")
    spec = concepts.parse_rule_edit(_load_json_from_text(await _post_to_fal_text(request)), num_train, num_test)
    if concepts.similar_to_recent("edit", spec.rule_spec, _concept_similarity):
        raise ValueError("Generated edit rule is too similar to a recent task")
    concepts.RECENT_CONCEPTS["edit"].append(spec.rule_spec)
    return spec


def pick_image_synth_category() -> str:
    categories = list(synth_cst.IMAGE_SYNTH_CATEGORY_WEIGHTS.keys())
    weights = list(synth_cst.IMAGE_SYNTH_CATEGORY_WEIGHTS.values())
    return random.choices(categories, weights=weights, k=1)[0]


def pick_image_training_mode(model_type: ImageModelType) -> ImageTrainingMode:
    capabilities = IMAGE_MODEL_CAPABILITIES[model_type]
    if capabilities == frozenset({ImageTrainingMode.EDIT}):
        return ImageTrainingMode.EDIT
    if ImageTrainingMode.EDIT in capabilities and random.random() < synth_cst.IMAGE_EDIT_TASK_PROBABILITY:
        return ImageTrainingMode.EDIT
    return ImageTrainingMode.TEXT_TO_IMAGE


async def generate_image_synthetic_by_category(
    config: Config, num_prompts: int, category: str, num_test: int = 0
) -> tuple[list[ImageTextPair], str, str | None]:
    """num_prompts training pairs, then num_test held-out pairs generated last (the task's test split)."""
    logger.info(f"Selected image synth category: {category}")

    if category in concepts.T2I_KINDS:
        return await generate_concept_synthetic(category, num_prompts, num_test)
    if category != synth_cst.IMAGE_SYNTH_CATEGORY_PERSON:
        raise ValueError(f"Unknown image synth category: {category}")

    last_result: tuple[list[ImageTextPair], str, str | None] | None = None
    for attempt in range(synth_cst.PERSON_GEN_RETRIES):
        image_text_pairs, ds_prefix, trigger_word = await generate_person_synthetic(num_prompts, num_test)
        last_result = (image_text_pairs, ds_prefix, trigger_word)
        if len(image_text_pairs) >= synth_cst.MIN_IMAGE_SYNTH_PAIRS + num_test:
            return image_text_pairs, ds_prefix, trigger_word
        if attempt < synth_cst.PERSON_GEN_RETRIES - 1:
            logger.info(f"Person synth generation only produced {len(image_text_pairs)} pairs, trying again...")
        else:
            logger.warning(
                f"Person synth generation only produced {len(image_text_pairs)} pairs after "
                f"{synth_cst.PERSON_GEN_RETRIES} attempts"
            )
    if last_result:
        return last_result
    raise ValueError("PERSON_GEN_RETRIES must be greater than zero")


def _image_competition_hours_for_dataset_size(num_images: int) -> float:
    """Scale competition length by dataset size in 15-minute steps."""
    min_images = synth_cst.MIN_IMAGE_SYNTH_PAIRS
    max_images = synth_cst.MAX_IMAGE_SYNTH_PAIRS
    min_hours = synth_cst.MIN_IMAGE_COMPETITION_HOURS
    max_hours = synth_cst.MAX_IMAGE_COMPETITION_HOURS

    if max_images <= min_images:
        return min_hours

    clamped_images = min(max(num_images, min_images), max_images)
    scale = (clamped_images - min_images) / (max_images - min_images)
    hours = min_hours + scale * (max_hours - min_hours)
    return math.ceil(hours * 4) / 4.0


async def create_synthetic_image_task(
    config: Config,
    models: AsyncGenerator[ImageModelInfo, None],
    *,
    max_num_prompts: int | None = None,
    training_mode: ImageTrainingMode | None = None,
) -> RawTask:
    """Create a synthetic image task with a random image dataset category."""
    logger.info("Creating synthetic image task")
    prompt_cap = min(max_num_prompts or synth_cst.MAX_IMAGE_SYNTH_PAIRS, synth_cst.MAX_IMAGE_SYNTH_PAIRS)
    if prompt_cap < synth_cst.MIN_IMAGE_SYNTH_PAIRS:
        raise ValueError(
            f"max_num_prompts must be at least {synth_cst.MIN_IMAGE_SYNTH_PAIRS}, got {prompt_cap}"
        )
    # Training pairs as before, plus a fixed held-out split generated on top so the test set grows without
    # shrinking the training set. Held-out pairs are written to leave the training distribution and come last.
    num_prompts = random.randint(synth_cst.MIN_IMAGE_SYNTH_PAIRS, prompt_cap)
    num_test = synth_cst.IMAGE_SYNTH_TEST_PAIRS
    model_info = await anext(models)
    if training_mode is None:
        training_mode = pick_image_training_mode(model_info.model_type)
    elif training_mode not in IMAGE_MODEL_CAPABILITIES.get(model_info.model_type, frozenset()):
        raise ValueError(f"{model_info.model_type.value} does not support {training_mode.value}")
    # Qwen families and MOSS edit need the +0.5h bump: large TEs / uncached VL encode.
    use_higher_training_hours = model_info.model_type in {
        ImageModelType.QWEN_IMAGE,
        ImageModelType.QWEN_IMAGE_EDIT_2511,
    } or (
        model_info.model_type == ImageModelType.MOSS_IMAGE_BASE
        and training_mode == ImageTrainingMode.EDIT
    )
    Path(TEMP_PATH_FOR_IMAGES).mkdir(parents=True, exist_ok=True)
    if training_mode == ImageTrainingMode.EDIT:
        image_text_pairs, ds_prefix, trigger_word = await generate_edit_synthetic(num_prompts, num_test)
    else:
        image_text_pairs, ds_prefix, trigger_word = await generate_image_synthetic_by_category(
            config, num_prompts, pick_image_synth_category(), num_test
        )
    if model_info.model_type == ImageModelType.IDEOGRAM4:
        image_text_pairs = await rewrite_pairs_with_ideogram_json_prompts(image_text_pairs)

    # Log image and text URLs for testing
    logger.info(f"Generated {len(image_text_pairs)} image-text pairs with prefix: {ds_prefix}")
    for i, pair in enumerate(image_text_pairs):
        logger.info(f"Pair {i+1} - Image URL: {pair.image_url}, Text URL: {pair.text_url}")

    training_pairs = len(image_text_pairs) - synth_cst.IMAGE_SYNTH_TEST_PAIRS
    if training_pairs >= synth_cst.MIN_IMAGE_SYNTH_PAIRS:
        number_of_hours = _image_competition_hours_for_dataset_size(training_pairs)
        if use_higher_training_hours:
            number_of_hours = round(number_of_hours + synth_cst.QWEN_IMAGE_EXTRA_COMPETITION_HOURS, 2)

        augmentation_config = maybe_get_augmentation_config(TaskType.IMAGETASK)
        task = ImageRawTask(
            model_id=model_info.model_id,
            ds=ds_prefix.replace(" ", "_").lower() + "_" + str(uuid.uuid4()),
            image_text_pairs=image_text_pairs,
            status=TaskStatus.PENDING,
            is_organic=False,
            created_at=datetime.utcnow(),
            termination_at=datetime.utcnow() + timedelta(hours=number_of_hours),
            hours_to_complete=number_of_hours,
            account_id=NULL_ACCOUNT_ID,
            model_type=model_info.model_type,
            training_mode=training_mode,
            trigger_word=trigger_word,
            augmentation_config=augmentation_config,
        )

        logger.info(f"New task created and added to the queue {task}")
        task = await add_task(task, config.psql_db)
        return task
    else:
        logger.error("Failed to generate enough image-text pairs for the task.")
        raise ValueError("Failed to generate enough image-text pairs for the task.")
