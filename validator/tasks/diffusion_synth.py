import asyncio
import json
import random
import re
import tempfile
import uuid
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import AsyncGenerator

from fiber import Keypair

import validator.core.constants as cst
from core.models.payload_models import ImageModelInfo
from core.models.payload_models import ImageTextPair
from core.models.utility_models import ImageModelType
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.core.models import ImageRawTask
from validator.core.models import RawTask
from validator.db.sql.tasks import add_task
from validator.utils.augmentation_decision import maybe_get_augmentation_config
from validator.utils.fal import download_url
from validator.utils.fal import extract_image_urls
from validator.utils.fal import extract_text
from validator.utils.fal import persist_image_text_pairs
from validator.utils.fal import post_to_fal
from validator.utils.fal import upload_local_file
from validator.utils.logging import get_logger
from validator.utils.util import retry_with_backoff


logger = get_logger(__name__)


IMAGE_STYLES = [
    "Watercolor Painting",
    "Oil Painting",
    "Digital Art",
    "Pencil Sketch",
    "Comic Book Style",
    "Cyberpunk",
    "Steampunk",
    "Impressionist",
    "Pop Art",
    "Minimalist",
    "Gothic",
    "Art Nouveau",
    "Pixel Art",
    "Anime",
    "3D Render",
    "Low Poly",
    "Photorealistic",
    "Vector Art",
    "Abstract Expressionism",
    "Realism",
    "Futurism",
    "Cubism",
    "Surrealism",
    "Baroque",
    "Renaissance",
    "Fantasy Illustration",
    "Sci-Fi Illustration",
    "Ukiyo-e",
    "Line Art",
    "Black and White Ink Drawing",
    "Graffiti Art",
    "Stencil Art",
    "Flat Design",
    "Isometric Art",
    "Retro 80s Style",
    "Vaporwave",
    "Dreamlike",
    "High Fantasy",
    "Dark Fantasy",
    "Medieval Art",
    "Art Deco",
    "Hyperrealism",
    "Sculpture Art",
    "Caricature",
    "Chibi",
    "Noir Style",
    "Lowbrow Art",
    "Psychedelic Art",
    "Vintage Poster",
    "Manga",
    "Holographic",
    "Kawaii",
    "Monochrome",
    "Geometric Art",
    "Photocollage",
    "Mixed Media",
    "Ink Wash Painting",
    "Charcoal Drawing",
    "Concept Art",
    "Digital Matte Painting",
    "Pointillism",
    "Expressionism",
    "Sumi-e",
    "Retro Futurism",
    "Pixelated Glitch Art",
    "Neon Glow",
    "Street Art",
    "Acrylic Painting",
    "Bauhaus",
    "Flat Cartoon Style",
    "Carved Relief Art",
    "Fantasy Realism",
]

with open(cst.EXAMPLE_PROMPTS_PATH, "r") as f:
    FULL_PROMPTS = json.load(f)


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


def create_image_style_compatibility_prompt(first_style: str, second_style: str) -> str:
    return f"""You are an expert in spotting incompatible artistic styles for image generation.
Analyze the {first_style} and {second_style} styles and determine if they can be effectively combined.
The styles are meant to be combined in a set of image generation prompts, so visual coherence is crucial.
Return only a JSON with a boolean 'compatible' field.

Example Output:
{{"compatible": true}}"""


def create_combined_diffusion_prompt(first_style: str, second_style: str, num_prompts: int) -> str:
    return f"""You are an expert in creating diverse and descriptive prompts for image generation models.
Generate {num_prompts} prompts in {first_style} and {second_style} style.

Requirements:
- Each prompt must clearly communicate the {first_style} and {second_style}'s distinctive visual characteristics
- Include specific visual elements that define this style (textures, colors, techniques)
- You MUST mention both of the chosen styles in the prompt
- Vary subject matter while maintaining style consistency
- Get super creative and do not repeat similar prompts
- The generated images should have a coherent style
- Return JSON only: {{"prompts": ["prompt 1", "prompt 2"]}}"""


def create_single_style_diffusion_prompt(style: str, num_prompts: int) -> str:
    prompt_examples = ",\n    ".join([f'"{prompt}"' for prompt in random.sample(FULL_PROMPTS[style], 5)])

    return f"""You are an expert in creating diverse and descriptive prompts for image generation models.
Generate {num_prompts} prompts in {style} style.

Here are examples of prompts in the {style} style. Follow the same quality bar, but do not copy them:
{{
"prompts": [
    {prompt_examples}
]
}}

Requirements:
- Each prompt must clearly communicate the {style}'s distinctive visual characteristics
- Include specific visual elements that define this style (textures, colors, techniques)
- You MUST mention the style in the prompt
- Vary subject matter while maintaining style consistency
- Get super creative and do not repeat similar prompts
- The generated images should have a coherent style
- Return JSON only: {{"prompts": ["prompt 1", "prompt 2"]}}"""


async def _post_to_fal_text(prompt: str) -> str:
    result = await post_to_fal(cst.FAL_TEXT_PROMPT_MODEL, {"prompt": prompt, "model": cst.FAL_TEXT_PROMPT_LLM})
    return extract_text(result)


@retry_with_backoff
async def generate_diffusion_prompts(first_style: str, second_style: str | None, keypair: Keypair, num_prompts: int) -> list[str]:
    if second_style:
        prompt = create_combined_diffusion_prompt(first_style, second_style, num_prompts)
        style_description = f"{first_style} and {second_style}"
    else:
        prompt = create_single_style_diffusion_prompt(first_style, num_prompts)
        style_description = first_style

    logger.info(f"Calling FAL text prompt model for {style_description}")
    result = await _post_to_fal_text(prompt)

    try:
        if isinstance(result, str):
            json_match = re.search(r"\{[\s\S]*\}", result)
            if json_match:
                logger.info(f"Full result from prompt generation for {style_description}: {result}")
                result = json_match.group(0)
            else:
                raise ValueError("Failed to generate a valid json")

        result_dict = json.loads(result) if isinstance(result, str) else result
        return result_dict["prompts"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to generate valid diffusion prompts: {e}")


async def check_style_compatibility(first_style: str, second_style: str, config: Config) -> bool:
    result = await _post_to_fal_text(create_image_style_compatibility_prompt(first_style, second_style))
    result_dict = json.loads(result) if isinstance(result, str) else result
    return result_dict.get("compatible", False)


async def pick_style_combination(config: Config) -> tuple[str, str]:
    for i in range(cst.IMAGE_STYLE_PICKING_NUM_TRIES):
        logger.info(f"Picking style combination. Try {i + 1} of {cst.IMAGE_STYLE_PICKING_NUM_TRIES}")
        first_style, second_style = random.sample(IMAGE_STYLES, 2)
        try:
            compatible = await check_style_compatibility(first_style, second_style, config)

            if compatible:
                return first_style, second_style
            logger.info(f"Styles {first_style} and {second_style} were found incompatible, trying new combination")
            continue

        except Exception as e:
            logger.error(f"Try {i + 1}/{cst.IMAGE_STYLE_PICKING_NUM_TRIES} failed: {e}")

    raise ValueError("Failed to pick a valid style combination")


async def _get_face_reference_url() -> str:
    Path(cst.TEMP_PATH_FOR_IMAGES).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cst.TEMP_PATH_FOR_IMAGES) as tmp_dir:
        reference_path = await download_url(cst.IMAGE_SYNTH_FACE_IMAGE_URL, Path(tmp_dir) / "face")
        return await upload_local_file(reference_path, "image_synth/face_refs")


def _person_prompt_request(num_prompts: int) -> str:
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
async def generate_person_prompts_with_fal_vision(face_image_url: str, num_prompts: int) -> list[str]:
    logger.info(f"Generating {num_prompts} person prompts with {cst.FAL_PERSON_PROMPT_MODEL}/{cst.FAL_PERSON_PROMPT_VLM}")
    payload = {
        "image_urls": [face_image_url],
        "prompt": _person_prompt_request(num_prompts),
        "model": cst.FAL_PERSON_PROMPT_VLM,
    }
    result = await post_to_fal(cst.FAL_PERSON_PROMPT_MODEL, payload)
    prompts = _get_prompts_from_response(extract_text(result), num_prompts)
    logger.info(f"Generated {len(prompts)} person prompts")
    return prompts


def _fal_image_payload(model_id: str, prompt: str, reference_image_url: str | None = None) -> dict:
    if model_id == cst.FAL_AVATAR_MODEL:
        if not reference_image_url:
            raise ValueError("reference_image_url is required for avatar generation")
        return {
            "prompt": prompt,
            "image_urls": [reference_image_url],
            "num_images": 1,
            "resolution": cst.FAL_NANO_BANANA_RESOLUTION,
            "output_format": cst.FAL_IMAGE_OUTPUT_FORMAT,
        }

    if model_id == cst.FAL_STYLE_MODEL_GPT_IMAGE_2:
        return {
            "prompt": prompt,
            "quality": cst.FAL_GPT_IMAGE_2_QUALITY,
            "num_images": 1,
            "output_format": cst.FAL_IMAGE_OUTPUT_FORMAT,
        }

    return {
        "prompt": prompt,
        "num_images": 1,
        "resolution": cst.FAL_NANO_BANANA_RESOLUTION,
        "output_format": cst.FAL_IMAGE_OUTPUT_FORMAT,
    }


@retry_with_backoff
async def generate_fal_image(model_id: str, prompt: str, reference_image_url: str | None = None) -> str:
    result = await post_to_fal(model_id, _fal_image_payload(model_id, prompt, reference_image_url))
    return extract_image_urls(result)[0]


async def generate_fal_images_for_prompts(
    model_id: str, prompts: list[str], reference_image_url: str | None = None
) -> list[tuple[str, str]]:
    semaphore = asyncio.Semaphore(cst.FAL_IMAGE_GENERATION_CONCURRENCY)

    async def generate_one(index: int, prompt: str) -> tuple[str, str]:
        async with semaphore:
            logger.info(f"Generating image {index + 1}/{len(prompts)} with {model_id}")
            image_url = await generate_fal_image(model_id, prompt, reference_image_url)
            logger.info(f"Generated image {index + 1}/{len(prompts)} with {model_id}")
            return image_url, prompt

    logger.info(
        f"Generating {len(prompts)} images with {model_id}, concurrency={cst.FAL_IMAGE_GENERATION_CONCURRENCY}"
    )
    return await asyncio.gather(*(generate_one(index, prompt) for index, prompt in enumerate(prompts)))


async def generate_style_synthetic(config: Config, num_prompts: int) -> tuple[list[ImageTextPair], str]:
    use_combined_styles = random.random() < cst.PROBABILITY_STYLE_COMBINATION

    if use_combined_styles:
        first_style, second_style = await pick_style_combination(config)
        logger.info(f"Picked style combination: {first_style} and {second_style}")
        ds_prefix = f"{first_style}_and_{second_style}"
    else:
        first_style = random.choice(IMAGE_STYLES)
        second_style = None
        logger.info(f"Picked style: {first_style}")
        ds_prefix = first_style

    try:
        logger.info(f"Generating {num_prompts} style prompts for {ds_prefix}")
        prompts = await generate_diffusion_prompts(first_style, second_style, config.keypair, num_prompts)
        logger.info(f"Generated {len(prompts)} style prompts for {ds_prefix}")
    except Exception as e:
        logger.error(f"Failed to generate prompts for {first_style} and {second_style}: {e}")
        raise e

    model_id = random.choice(cst.FAL_STYLE_MODELS)
    logger.info(f"Selected FAL style model for full task: {model_id}")
    image_prompt_pairs = await generate_fal_images_for_prompts(model_id, prompts)
    logger.info(f"Persisting {len(image_prompt_pairs)} style image-text pairs")
    image_text_pairs = await persist_image_text_pairs(image_prompt_pairs)
    logger.info(f"Persisted {len(image_text_pairs)} style image-text pairs")
    return image_text_pairs, ds_prefix


async def generate_person_synthetic(num_prompts: int) -> tuple[list[ImageTextPair], str]:
    logger.info("Fetching and uploading person reference image")
    face_image_url = await _get_face_reference_url()
    prompts = await generate_person_prompts_with_fal_vision(face_image_url, num_prompts)
    image_prompt_pairs = await generate_fal_images_for_prompts(cst.FAL_AVATAR_MODEL, prompts, face_image_url)
    logger.info(f"Persisting {len(image_prompt_pairs)} person image-text pairs")
    image_text_pairs = await persist_image_text_pairs(image_prompt_pairs)
    logger.info(f"Persisted {len(image_text_pairs)} person image-text pairs")
    return image_text_pairs, cst.PERSON_SYNTH_DS_PREFIX


def _random_image_competition_hours() -> float:
    """Pick competition length in 15-minute (0.25h) steps between min and max."""
    min_q = int(round(cst.MIN_IMAGE_COMPETITION_HOURS * 4))
    max_q = int(round(cst.MAX_IMAGE_COMPETITION_HOURS * 4))
    return random.randint(min_q, max_q) / 4.0


async def create_synthetic_image_task(config: Config, models: AsyncGenerator[ImageModelInfo, None]) -> RawTask:
    """Create a synthetic image task with random model and style."""
    logger.info("Creating synthetic image task")
    number_of_hours = _random_image_competition_hours()
    num_prompts = random.randint(cst.MIN_IMAGE_SYNTH_PAIRS, cst.MAX_IMAGE_SYNTH_PAIRS)
    model_info = await anext(models)
    is_qwen_model = model_info.model_type == ImageModelType.QWEN_IMAGE
    if is_qwen_model:
        number_of_hours = round(number_of_hours + cst.QWEN_IMAGE_EXTRA_COMPETITION_HOURS, 2)
    Path(cst.TEMP_PATH_FOR_IMAGES).mkdir(parents=True, exist_ok=True)
    if random.random() < cst.PERCENTAGE_OF_IMAGE_SYNTHS_SHOULD_BE_STYLE:
        image_text_pairs, ds_prefix = await generate_style_synthetic(config, num_prompts)
    else:
        # Try person synth with a few retries for insufficient pairs
        for attempt in range(cst.PERSON_GEN_RETRIES):
            image_text_pairs, ds_prefix = await generate_person_synthetic(num_prompts)
            if len(image_text_pairs) >= 10:
                break
            elif attempt < cst.PERSON_GEN_RETRIES - 1:
                logger.info(f"Person synth generation only produced {len(image_text_pairs)} pairs, trying again...")
            else:
                logger.warning(
                    f"Person synth generation only produced {len(image_text_pairs)} pairs after {cst.PERSON_GEN_RETRIES} attempts"
                )

    # Log image and text URLs for testing
    logger.info(f"Generated {len(image_text_pairs)} image-text pairs with prefix: {ds_prefix}")
    for i, pair in enumerate(image_text_pairs):
        logger.info(f"Pair {i+1} - Image URL: {pair.image_url}, Text URL: {pair.text_url}")

    if len(image_text_pairs) >= 10:
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
            account_id=cst.NULL_ACCOUNT_ID,
            model_type=model_info.model_type,
            augmentation_config=augmentation_config,
        )

        logger.info(f"New task created and added to the queue {task}")
        task = await add_task(task, config.psql_db)
        return task
    else:
        logger.error("Failed to generate enough image-text pairs for the task.")
        raise ValueError("Failed to generate enough image-text pairs for the task.")
