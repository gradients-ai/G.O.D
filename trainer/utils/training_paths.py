import os
from pathlib import Path

import trainer.constants as train_cst
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import EnvironmentDatasetType


def get_checkpoints_output_path(task_id: str, repo_name: str) -> str:
    return str(Path(train_cst.OUTPUT_CHECKPOINTS_PATH) / task_id / repo_name)

def get_image_base_model_path(model_id: str) -> str:
    model_folder = model_id.replace("/", "--")
    base_path = str(Path(train_cst.CACHE_MODELS_DIR) / model_folder)
    if os.path.isdir(base_path):
        files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(base_path, files[0])
    return base_path

def get_image_training_images_dir(task_id: str) -> str:
    return str(Path(train_cst.IMAGE_CONTAINER_IMAGES_PATH) / task_id / "img")

def get_image_training_config_template_path(model_type: str) -> str:
    model_type = model_type.lower()
    if model_type == ImageModelType.SDXL.value:
        return str(Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / "base_diffusion_sdxl.toml")
    elif model_type == ImageModelType.FLUX.value:
        return str(Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / "base_diffusion_flux.toml")
    elif model_type == ImageModelType.Z_IMAGE.value:
        return str(Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / "base_diffusion_zimage.yaml")
    elif model_type == ImageModelType.QWEN_IMAGE.value:
        return str(Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / "base_diffusion_qwen_image.yaml")

def get_image_training_zip_save_path(task_id: str) -> str:
    return str(Path(train_cst.CACHE_DATASETS_DIR) / f"{task_id}_tourn.zip")

def get_text_dataset_path(task_id: str) -> str:
    return str(Path(train_cst.CACHE_DATASETS_DIR) / f"{task_id}_train_data.json")

def get_axolotl_dataset_paths(dataset_filename: str) -> tuple[str, str]:
    data_path = str(Path(train_cst.AXOLOTL_DIRECTORIES["data"]) / dataset_filename)
    root_path = str(Path(train_cst.AXOLOTL_DIRECTORIES["root"]) / dataset_filename)
    return data_path, root_path

def get_axolotl_base_config_path(dataset_type) -> str:
    root_dir = Path(train_cst.AXOLOTL_DIRECTORIES["root"])
    if isinstance(dataset_type, EnvironmentDatasetType):
        return str(root_dir / "base_environment.yml")
    elif isinstance(dataset_type, (InstructTextDatasetType, DpoDatasetType, ChatTemplateDatasetType)):
        return str(root_dir / "base.yml")
    elif isinstance(dataset_type, GrpoDatasetType):
        return str(root_dir / "base_grpo.yml")
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")

def get_text_base_model_path(model_id: str) -> str:
    model_folder = model_id.replace("/", "--")
    return str(Path(train_cst.CACHE_MODELS_DIR) / model_folder)