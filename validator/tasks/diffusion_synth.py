import asyncio
import json
import os
import random
import re
import shutil
import tempfile
import uuid
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import AsyncGenerator

import docker
from fiber import Keypair

import validator.core.constants as cst
from core.models.payload_models import ImageModelInfo
from core.models.payload_models import ImageTextPair
from core.models.utility_models import ImageModelType
from core.models.utility_models import Message
from core.models.utility_models import Role
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.models import ImageRawTask
from validator.core.models import RawTask
from validator.db.sql.tasks import add_task
from validator.tasks.task_prep import upload_file_to_minio
from validator.utils.call_endpoint import post_to_nineteen_image
from validator.utils.llm import convert_to_nineteen_payload
from validator.utils.llm import post_to_nineteen_chat_with_reasoning
from validator.utils.logging import get_all_context_tags
from validator.utils.logging import get_logger
from validator.utils.logging import stream_container_logs
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


def create_image_style_compatibility_messages(first_style: str, second_style: str) -> list[Message]:
    system_content = """You are an expert in spotting incompatible artistic styles for image generation.
Your task is to analyze two artistic styles and determine if they are not compatible to be merged into a style that has both characteristics.
The styles are meant to be combined in a set of prompts for image generation.
It is crucial for the generated images to have a coherent style."""

    user_content = f"""Analyze the {first_style} and {second_style} styles and determine if they can be effectively combined.
Return only a JSON with a boolean 'compatible' field.

Example Output:
{{"compatible": true}}"""

    return [Message(role=Role.SYSTEM, content=system_content), Message(role=Role.USER, content=user_content)]


def create_combined_diffusion_messages(first_style: str, second_style: str, num_prompts: int) -> list[Message]:
    system_content = """You are an expert in creating diverse and descriptive prompts for image generation models.
    Your task is to generate prompts that strongly embody a combination of two artistic styles.
    Each prompt should be detailed and consistent with both of the given styles.
    You will return the prompts in a JSON format with no additional text.
    """

    user_content = f"""Generate {num_prompts} prompts in {first_style} and {second_style} style.

    Requirements:
    - Each prompt must clearly communicate the {first_style} and {second_style}'s distinctive visual characteristics
    - Include specific visual elements that define this style (textures, colors, techniques)
    - You MUST mention both of the chosen styles in the prompt
    - Vary subject matter while maintaining style consistency
    - Get super creative and do not repeat similar prompts!
    - The generated images should have a coherent style
    - Return JSON only"""

    return [Message(role=Role.SYSTEM, content=system_content), Message(role=Role.USER, content=user_content)]


def create_single_style_diffusion_messages(style: str, num_prompts: int) -> list[Message]:
    prompt_examples = ",\n    ".join([f'"{prompt}"' for prompt in random.sample(FULL_PROMPTS[style], 5)])

    system_content = f"""You are an expert in creating diverse and descriptive prompts for image generation models.
    Your task is to generate prompts that strongly embody a combination of an artistic style.
    Each prompt should be detailed and consistent with the given style.
    You will return the prompts in a JSON format with no additional text.

    Here are some examples of prompts in the {style} style, you need to follow the same format and generate more in the same style:
    {{
    "prompts": [
        {prompt_examples}
    ]
    }}"""

    user_content = f"""Generate {num_prompts} prompts in {style} style.

    Requirements:
    - Each prompt must clearly communicate the {style}'s distinctive visual characteristics
    - Include specific visual elements that define this style (textures, colors, techniques)
    - You MUST mention the style in the prompt
    - Vary subject matter while maintaining style consistency
    - Get super creative and do not repeat similar prompts!
    - The generated images should have a coherent style
    - Return JSON only"""

    return [Message(role=Role.SYSTEM, content=system_content), Message(role=Role.USER, content=user_content)]


@retry_with_backoff
async def generate_diffusion_prompts(first_style: str, second_style: str | None, keypair: Keypair, num_prompts: int) -> list[str]:
    if second_style:
        messages = create_combined_diffusion_messages(first_style, second_style, num_prompts)
        style_description = f"{first_style} and {second_style}"
    else:
        messages = create_single_style_diffusion_messages(first_style, num_prompts)
        style_description = first_style

    payload = convert_to_nineteen_payload(
        messages, cst.IMAGE_PROMPT_GEN_MODEL, cst.IMAGE_PROMPT_GEN_MODEL_TEMPERATURE, cst.IMAGE_PROMPT_GEN_MODEL_MAX_TOKENS
    )

    result = await post_to_nineteen_chat_with_reasoning(payload, keypair, cst.END_OF_REASONING_TAG)

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


@retry_with_backoff
async def generate_image(prompt: str, keypair: Keypair, width: int, height: int) -> str:
    """Generate an image using the Nineteen AI API.

    Args:
        prompt: The text prompt to generate an image from
        keypair: The keypair containing the API key
        width: The width in pixels of the image to generate
        height: The height in pixels of the image to generate

    Returns:
        str: The base64-encoded image data
    """
    payload = {
        "prompt": prompt,
        "model": cst.IMAGE_GEN_MODEL,
        "num_inference_steps": cst.IMAGE_GEN_STEPS,
        "guidance_scale": cst.IMAGE_GEN_CFG_SCALE,
        "height": height,
        "width": width,
        "negative_prompt": "",
    }

    result = await post_to_nineteen_image(payload, keypair)

    try:
        image_bytes = result.content
        return image_bytes
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error parsing image generation response: {e}")
        raise ValueError("Failed to generate image")


async def check_style_compatibility(first_style: str, second_style: str, config: Config) -> bool:
    messages = create_image_style_compatibility_messages(first_style, second_style)
    payload = convert_to_nineteen_payload(messages, cst.IMAGE_PROMPT_GEN_MODEL, cst.IMAGE_PROMPT_GEN_MODEL_TEMPERATURE)
    result = await post_to_nineteen_chat_with_reasoning(payload, config.keypair, cst.END_OF_REASONING_TAG)
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
        prompts = await generate_diffusion_prompts(first_style, second_style, config.keypair, num_prompts)
    except Exception as e:
        logger.error(f"Failed to generate prompts for {first_style} and {second_style}: {e}")
        raise e

    client = docker.from_env()
    image_text_pairs = []
    with tempfile.TemporaryDirectory(dir=cst.TEMP_PATH_FOR_IMAGES) as tmp_dir_path:
        container = await asyncio.to_thread(
            client.containers.run,
            image=cst.IMAGE_SYNTH_DOCKER_IMAGE,
            environment={
                "SAVE_DIR": cst.SYNTH_CONTAINER_SAVE_PATH,
                "PROMPTS": json.dumps(prompts),
            },
            volumes={tmp_dir_path: {"bind": cst.SYNTH_CONTAINER_SAVE_PATH, "mode": "rw"}},
            device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=["0"])],
            detach=True,
        )
        log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))
        result = await asyncio.to_thread(container.wait)
        log_task.cancel()
        images_dir = Path(tmp_dir_path)
        for file in images_dir.iterdir():
            if file.is_file() and file.suffix == ".png":
                txt_path = images_dir / f"{file.stem}.txt"
                if txt_path.exists() and txt_path.stat().st_size > 0:
                    img_url = await upload_file_to_minio(str(file), cst.BUCKET_NAME, f"{os.urandom(8).hex()}.png")
                    txt_url = await upload_file_to_minio(str(txt_path), cst.BUCKET_NAME, f"{os.urandom(8).hex()}.txt")
                    image_text_pairs.append(ImageTextPair(image_url=img_url, text_url=txt_url))
        if os.path.exists(tmp_dir_path):
            shutil.rmtree(tmp_dir_path)

    await asyncio.to_thread(client.containers.prune)
    await asyncio.to_thread(client.images.prune, filters={"dangling": True})
    await asyncio.to_thread(client.volumes.prune)

    return image_text_pairs, ds_prefix


async def generate_person_synthetic(num_prompts: int) -> tuple[list[ImageTextPair], str]:
    client = docker.from_env()
    image_text_pairs = []
    with tempfile.TemporaryDirectory(dir=cst.TEMP_PATH_FOR_IMAGES) as tmp_dir_path:
        container = await asyncio.to_thread(
            client.containers.run,
            image=cst.IMAGE_SYNTH_DOCKER_IMAGE,
            environment={"SAVE_DIR": cst.SYNTH_CONTAINER_SAVE_PATH, "NUM_PROMPTS": num_prompts},
            volumes={tmp_dir_path: {"bind": cst.SYNTH_CONTAINER_SAVE_PATH, "mode": "rw"}},
            device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=["0"])],
            detach=True,
        )
        log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))
        result = await asyncio.to_thread(container.wait)
        log_task.cancel()
        images_dir = Path(tmp_dir_path)
        for file in images_dir.iterdir():
            if file.is_file() and file.suffix == ".png":
                txt_path = images_dir / f"{file.stem}.txt"
                if txt_path.exists() and txt_path.stat().st_size > 0:
                    img_url = await upload_file_to_minio(str(file), cst.BUCKET_NAME, f"{os.urandom(8).hex()}.png")
                    txt_url = await upload_file_to_minio(str(txt_path), cst.BUCKET_NAME, f"{os.urandom(8).hex()}.txt")
                    image_text_pairs.append(ImageTextPair(image_url=img_url, text_url=txt_url))
        if os.path.exists(tmp_dir_path):
            shutil.rmtree(tmp_dir_path)

    await asyncio.to_thread(client.containers.prune)
    await asyncio.to_thread(client.images.prune, filters={"dangling": True})
    await asyncio.to_thread(client.volumes.prune)

    return image_text_pairs, cst.PERSON_SYNTH_DS_PREFIX


async def create_synthetic_image_task(config: Config, models: AsyncGenerator[ImageModelInfo, None]) -> RawTask:
    """Create a synthetic image task with random model and style."""
    logger.info("Creating synthetic image task")
    number_of_hours = random.randint(cst.MIN_IMAGE_COMPETITION_HOURS, cst.MAX_IMAGE_COMPETITION_HOURS)
    num_prompts = random.randint(cst.MIN_IMAGE_SYNTH_PAIRS, cst.MAX_IMAGE_SYNTH_PAIRS)
    model_info = await anext(models)
    is_qwen_model = model_info.model_type == ImageModelType.QWEN_IMAGE
    if is_qwen_model:
        number_of_hours += 1
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
        )

        logger.info(f"New task created and added to the queue {task}")
        task = await add_task(task, config.psql_db)
        return task
    else:
        logger.error("Failed to generate enough image-text pairs for the task.")
        raise ValueError("Failed to generate enough image-text pairs for the task.")
