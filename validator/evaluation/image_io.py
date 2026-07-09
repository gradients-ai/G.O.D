import base64
import os
import shutil
import tempfile
from io import BytesIO

from huggingface_hub import hf_hub_download
from PIL import Image

from core.logging import get_logger


logger = get_logger(__name__)


def adjust_image_size(image: Image.Image) -> Image.Image:
    width, height = image.size

    if width > height:
        new_width = 1024
        new_height = int((height / width) * 1024)
    else:
        new_height = 1024
        new_width = int((width / height) * 1024)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16

    width, height = image.size
    crop_width = min(width, new_width)
    crop_height = min(height, new_height)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    image = image.crop((left, top, right, bottom))

    return image

def base64_to_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)
    return image

def download_from_huggingface(repo_id: str, filename: str, local_dir: str) -> str:
    # Use a temp folder to ensure correct file placement
    try:
        local_filename = f"models--{repo_id.replace('/', '--')}.safetensors"
        final_path = os.path.join(local_dir, local_filename)
        if os.path.exists(final_path):
            logger.info(f"File {filename} already exists. Skipping download.")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=temp_dir)
                shutil.move(temp_file_path, final_path)
            logger.info(f"File {filename} downloaded successfully")
        return final_path
    except Exception as e:
        logger.error(f"Error downloading file: {e}")

def list_supported_images(dataset_path: str, extensions: tuple) -> list[str]:
    return [file_name for file_name in os.listdir(dataset_path) if file_name.lower().endswith(extensions)]

def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    img_format = image.format if image.format else "PNG"
    image.save(buffer, format=img_format)
    return base64.b64encode(buffer.getvalue()).decode()

def read_prompt_file(text_file_path: str) -> str:
    if os.path.exists(text_file_path):
        with open(text_file_path, "r", encoding="utf-8") as text_file:
            return text_file.read()
    return None
