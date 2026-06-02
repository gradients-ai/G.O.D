import os
import shutil
import zipfile

import trainer.constants as train_cst


def prepare_dataset(
    training_images_zip_path: str,
    training_images_repeat: int,
    instance_prompt: str,
    class_prompt: str,
    job_id: str,
    regularization_images_dir: str = None,
    regularization_images_repeat: int = None,
    output_dir: str = None,
):
    base_output_dir = output_dir or train_cst.IMAGE_CONTAINER_IMAGES_PATH
    extraction_dir = os.path.join(base_output_dir, "tmp", job_id)
    os.makedirs(extraction_dir, exist_ok=True)
    with zipfile.ZipFile(training_images_zip_path, "r") as zip_ref:
        zip_ref.extractall(extraction_dir)

    extracted_items = [entry for entry in os.listdir(extraction_dir)]
    if len(extracted_items) == 1 and os.path.isdir(os.path.join(extraction_dir, extracted_items[0])):
        training_images_dir = os.path.join(extraction_dir, extracted_items[0])
    else:
        training_images_dir = extraction_dir

    output_dir = os.path.join(base_output_dir, job_id)
    os.makedirs(output_dir, exist_ok=True)

    training_dir = os.path.join(
        output_dir,
        f"img/{training_images_repeat}_{instance_prompt} {class_prompt}",
    )

    if os.path.exists(training_dir):
        shutil.rmtree(training_dir)

    shutil.copytree(training_images_dir, training_dir)

    if regularization_images_dir is not None:
        regularization_dir = os.path.join(
            output_dir,
            f"reg/{regularization_images_repeat}_{class_prompt}",
        )

        if os.path.exists(regularization_dir):
            shutil.rmtree(regularization_dir)
        shutil.copytree(regularization_images_dir, regularization_dir)

    if not os.path.exists(os.path.join(output_dir, "log")):
        os.makedirs(os.path.join(output_dir, "log"))

    if not os.path.exists(os.path.join(output_dir, "model")):
        os.makedirs(os.path.join(output_dir, "model"))

    if os.path.exists(extraction_dir):
        shutil.rmtree(extraction_dir)

    if os.path.exists(training_images_zip_path) and "tourn" not in os.path.basename(training_images_zip_path):
        os.remove(training_images_zip_path)
