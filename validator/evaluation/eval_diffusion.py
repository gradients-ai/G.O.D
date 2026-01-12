import json
import os
import re
import random

import numpy as np
import safetensors.torch
from diffusers import StableDiffusionPipeline
from fiber.logging_utils import get_logger
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download
from PIL import Image

from core.models.utility_models import ImageModelType
from validator.core import constants as cst
from validator.core.models import Img2ImgPayload
from validator.evaluation.utils import adjust_image_size
from validator.evaluation.utils import base64_to_image
from validator.evaluation.utils import download_from_huggingface
from validator.evaluation.utils import image_to_base64
from validator.evaluation.utils import list_supported_images
from validator.evaluation.utils import read_prompt_file
from validator.utils import comfy_api_gate as api_gate
from validator.utils.retry_utils import retry_on_5xx


logger = get_logger(__name__)
hf_api = HfApi()


def generate_reproducible_seeds(master_seed: int, n: int = 10) -> list[int]:
    random.seed(master_seed) 
    return [random.randint(0, 2**32 - 1) for _ in range(n)]

def load_comfy_workflows(model_type: str):
    if model_type == ImageModelType.SDXL.value:
        with open(cst.LORA_SDXL_WORKFLOW_PATH, "r") as file:
            lora_template = json.load(file)

        with open(cst.LORA_SDXL_WORKFLOW_PATH_DIFFUSERS, "r") as file:
            lora_template_diffusers = json.load(file)

        return lora_template, lora_template_diffusers
    elif model_type == ImageModelType.FLUX.value:
        with open(cst.LORA_FLUX_WORKFLOW_PATH, "r") as file:
            lora_template = json.load(file)

        return lora_template, None
    elif model_type == ImageModelType.Z_IMAGE.value:
        with open(cst.LORA_ZIMAGE_WORKFLOW_PATH, "r") as file:
            lora_template = json.load(file)

        return lora_template, None
    elif model_type == ImageModelType.QWEN_IMAGE.value:
        with open(cst.LORA_QWEN_IMAGE_WORKFLOW_PATH, "r") as file:
            lora_template = json.load(file)

        return lora_template, None
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def contains_image_files(directory: str) -> str:
    try:
        return any(file.lower().endswith(cst.SUPPORTED_IMAGE_FILE_EXTENSIONS) for file in os.listdir(directory))
    except FileNotFoundError:
        return False


def validate_dataset_path(dataset_path: str) -> str:
    if os.path.isdir(dataset_path):
        if contains_image_files(dataset_path):
            return dataset_path
        subdirectories = [
            os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))
        ]
        for subdirectory in subdirectories:
            if contains_image_files(subdirectory):
                return subdirectory
    return dataset_path


@retry_on_5xx()
def find_latest_lora_submission_name(repo_id: str) -> str:
    repo_files = hf_api.list_repo_files(repo_id)
    model_files = [file for file in repo_files if file.startswith(cst.DIFFUSION_HF_DEFAULT_FOLDER)]

    for file in model_files:
        if file.endswith(cst.DIFFUSION_HF_DEFAULT_CKPT_NAME):
            return file

    epoch_files = []
    
    for file in model_files:
        if file.endswith(".safetensors"):
            epoch = None
            match = re.search(r'[-_](\d+)\.safetensors$', file)
            if match:
                try:
                    epoch = int(match.group(1))
                except ValueError:
                    pass
            else:
                match = re.search(r'(\d+)\.safetensors$', file)
                if match:
                    try:
                        epoch = int(match.group(1))
                    except ValueError:
                        pass
            
            if epoch is None:
                return file
            else:
                epoch_files.append((epoch, file))

    if epoch_files:
        epoch_files.sort(reverse=True, key=lambda x: x[0])
        return epoch_files[0][1]

    return None


@retry_on_5xx()
def is_safetensors_available(repo_id: str, model_type: str) -> tuple[bool, str | None]:    
    files_metadata = hf_api.list_repo_tree(repo_id=repo_id, repo_type="model")
    check_size_in_gb = 6 if model_type == "sdxl" else 10
    total_check_size = check_size_in_gb * 1024 * 1024 * 1024
    largest_file = None
    for file in files_metadata:
        if hasattr(file, "size") and file.size is not None:
            if file.path.endswith(".safetensors") and file.size > total_check_size:
                if largest_file is None or file.size > largest_file.size:
                    largest_file = file

    if largest_file:
        return True, largest_file.path
    return False, None


def download_base_model(repo_id: str, model_type: str, safetensors_filename: str | None = None) -> str:
    if model_type == ImageModelType.SDXL.value:
        download_dir = cst.CHECKPOINTS_SAVE_PATH
    elif model_type == ImageModelType.FLUX.value:
        download_dir = cst.UNET_SAVE_PATH
    else:
        download_dir = cst.DIFFUSION_MODELS_PATH

    if safetensors_filename:
        model_path = download_from_huggingface(repo_id, safetensors_filename, download_dir)
        model_name = os.path.basename(model_path)
    else:
        model_name = f"models--{repo_id.replace('/', '--')}"
        save_dir = f"{cst.DIFFUSERS_PATH}/{model_name}"
        model_path = snapshot_download(repo_id=repo_id, local_dir=save_dir, repo_type="model")
    return model_name, model_path


def download_lora(repo_id: str) -> str:
    lora_save_name = repo_id.split("/")[-1]
    if not os.path.exists(f"{cst.LORAS_SAVE_PATH}/{lora_save_name}.safetensors"):
        lora_filename = find_latest_lora_submission_name(repo_id)
        local_path = download_from_huggingface(repo_id, lora_filename, cst.LORAS_SAVE_PATH)
        unique_path = f"{cst.LORAS_SAVE_PATH}/{lora_save_name}.safetensors"
        os.rename(local_path, unique_path)
        logger.info(f"Downloaded {unique_path}")
        return unique_path
    else:
        return f"{cst.LORAS_SAVE_PATH}/{lora_save_name}.safetensors"


def calculate_l2_loss(test_image: Image.Image, generated_image: Image.Image) -> float:
    test_image = np.array(test_image.convert("RGB")) / 255.0
    generated_image = np.array(generated_image.convert("RGB")) / 255.0
    if test_image.shape != generated_image.shape:
        raise ValueError("Images must have the same dimensions to calculate L2 loss.")
    l2_loss = np.mean((test_image - generated_image) ** 2)
    return l2_loss


def edit_workflow(
    payload: dict, edit_elements: Img2ImgPayload, text_guided: bool, model_type: str, seed: int, is_safetensors: bool = True
) -> dict:
    if model_type == ImageModelType.SDXL.value:
        if is_safetensors:
            payload["Checkpoint_loader"]["inputs"]["ckpt_name"] = edit_elements.ckpt_name
        else:
            payload["Checkpoint_loader"]["inputs"]["model_path"] = edit_elements.ckpt_name
        payload["Sampler"]["inputs"]["cfg"] = edit_elements.cfg        
    elif model_type == ImageModelType.FLUX.value:
        payload["Checkpoint_loader"]["inputs"]["unet_name"] = edit_elements.ckpt_name
        payload["CFG"]["inputs"]["guidance"] = edit_elements.cfg
    else:
        payload["Checkpoint_loader"]["inputs"]["unet_name"] = edit_elements.ckpt_name
        payload["Sampler"]["inputs"]["cfg"] = edit_elements.cfg

    payload["Sampler"]["inputs"]["steps"] = edit_elements.steps
    payload["Sampler"]["inputs"]["seed"] = edit_elements.seed
    payload["Sampler"]["inputs"]["denoise"] = edit_elements.denoise
    payload["Image_loader"]["inputs"]["image"] = edit_elements.base_image
    payload["Lora_loader"]["inputs"]["lora_name"] = edit_elements.lora_name
    if text_guided:
        payload["Prompt"]["inputs"]["text"] = edit_elements.prompt
    else:
        payload["Prompt"]["inputs"]["text"] = ""

    return payload


def inference(image_base64: str, params: Img2ImgPayload, use_prompt: bool = False, prompt: str = None) -> tuple[float, float]:
    if use_prompt and prompt:
        params.prompt = prompt

    params.base_image = image_base64

    lora_payload = edit_workflow(
        payload=params.comfy_template,
        edit_elements=params,
        text_guided=use_prompt,
        model_type=params.model_type,
        seed=params.seed,
        is_safetensors=params.is_safetensors,
    )
    lora_gen = api_gate.generate(lora_payload)[0]
    lora_gen_loss = calculate_l2_loss(base64_to_image(image_base64), lora_gen)
    logger.info(f"Loss: {lora_gen_loss}")

    return lora_gen_loss


def eval_loop(dataset_path: str, params: Img2ImgPayload) -> dict[str, list[float]]:
    total_text_guided_losses = []
    total_no_text_losses = []

    test_images_list = list_supported_images(dataset_path, cst.SUPPORTED_IMAGE_FILE_EXTENSIONS)

    for file_name in test_images_list:
        logger.info(f"Calculating losses for {file_name}")

        base_name = os.path.splitext(file_name)[0]
        png_path = os.path.join(dataset_path, file_name)
        txt_path = os.path.join(dataset_path, f"{base_name}.txt")
        test_image = Image.open(png_path)
        test_image = adjust_image_size(test_image)
        image_base64 = image_to_base64(test_image)
        prompt = read_prompt_file(txt_path)

        params.prompt = prompt
        seeds = generate_reproducible_seeds(master_seed=42, n=10)
        text_guided_losses = []
        no_text_losses = []
        for seed in seeds:
            params.seed = seed
            text_guided_losses.append(inference(image_base64, params, use_prompt=True))
            no_text_losses.append(inference(image_base64, params, use_prompt=False))
        total_text_guided_losses.append(np.mean(text_guided_losses))
        total_no_text_losses.append(np.mean(no_text_losses))

    return {"text_guided_losses": total_text_guided_losses, "no_text_losses": total_no_text_losses}


def _count_model_parameters(model_path: str, is_safetensors: bool) -> int:
    try:
        if is_safetensors:
            state_dict = safetensors.torch.load_file(model_path)
            return sum(p.numel() for p in state_dict.values()) or 0
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_path)
            total_params = 0
            for attr in pipe.__dict__.values():
                if hasattr(attr, "parameters"):
                    total_params += sum(p.numel() for p in attr.parameters())
            return total_params
    except Exception as e:
        logger.error(f"Failed to count model parameters: {e}")
        return 0


def main():
    test_dataset_path = os.environ.get("DATASET")
    base_model_repo = os.environ.get("ORIGINAL_MODEL_REPO")
    trained_lora_model_repos = os.environ.get("MODELS", "")
    model_type = os.environ.get("MODEL_TYPE")
    if not all([test_dataset_path, base_model_repo, trained_lora_model_repos, model_type]):
        logger.error("Missing required environment variables.")
        exit(1)

    is_safetensors, safetensors_filename = is_safetensors_available(base_model_repo, model_type)
    # Base model download
    logger.info("Downloading base model")
    model_name_or_path, model_path = download_base_model(
        base_model_repo, model_type=model_type, safetensors_filename=safetensors_filename
    )
    logger.info("Base model downloaded")

    logger.info("test_dataset_path: ", test_dataset_path)
    logger.info("base_model_repo: ", base_model_repo)
    logger.info("trained_lora_model_repos: ", trained_lora_model_repos)
    logger.info("model_type: ", model_type)
    logger.info("is_safetensors: ", is_safetensors)
    logger.info("safetensors_filename: ", safetensors_filename)
    logger.info("model_name_or_path: ", model_name_or_path)
    logger.info("model_path: ", model_path)

    lora_repos = [m.strip() for m in trained_lora_model_repos.split(",") if m.strip()]

    test_dataset_path = validate_dataset_path(test_dataset_path)

    lora_comfy_template, diffusers_comfy_template = load_comfy_workflows(model_type)
    api_gate.connect()

    results = {"model_params_count": _count_model_parameters(model_path, is_safetensors)}

    generation_params = cst.EVAL_DEFAULTS.get(model_type, cst.EVAL_DEFAULTS[ImageModelType.SDXL.value])

    for repo_id in lora_repos:
        try:
            lora_local_path = download_lora(repo_id)
            img2img_payload = Img2ImgPayload(
                ckpt_name=model_name_or_path,
                lora_name=os.path.basename(lora_local_path),
                steps=generation_params["steps"],
                cfg=generation_params["cfg"],
                denoise=generation_params["denoise"],
                comfy_template=lora_comfy_template if is_safetensors else diffusers_comfy_template,
                is_safetensors=is_safetensors,
                model_type=model_type,
            )

            loss_data = eval_loop(test_dataset_path, img2img_payload)
            results[repo_id] = {"eval_loss": loss_data}

            if os.path.exists(lora_local_path):
                os.remove(lora_local_path)
        except Exception as e:
            logger.error(f"Error evaluating repo {repo_id}: {str(e)}")
            results[repo_id] = str(e)

    output_file = "/aplp/evaluation_results.json"
    output_dir = os.path.dirname(output_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w") as f:
        json.dump(results, f)

    logger.info(f"Evaluation results saved to {output_file}")

    logger.info(json.dumps(results))


if __name__ == "__main__":
    main()
