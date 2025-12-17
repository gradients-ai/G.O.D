import io
import json
import os
import random
import re
import uuid
from contextlib import redirect_stdout
from copy import deepcopy
from io import BytesIO

import names
import requests
from llava.eval.run_llava import eval_model
from llava.mm_utils import get_model_name_from_path
from PIL import Image

import validator.tasks.image_synth.constants as cst
import validator.utils.comfy_api_gate as api_gate


with open(cst.STYLE_WORKFLOW_PATH, "r") as file:
    style_template = json.load(file)

if __name__ == "__main__":
    prompts = json.loads(os.environ["PROMPTS"])
    
    api_gate.connect()
    save_dir = cst.DEFAULT_SAVE_DIR

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for prompt in prompts:
        workflow = deepcopy(style_template)
        workflow["Prompt"]["inputs"]["text"] += prompt
        image = api_gate.generate(workflow)[0]
        image_id = uuid.uuid4()
        image.save(f"{save_dir}{image_id}.png")
        with open(f"{save_dir}{image_id}.txt", "w") as file:
            file.write(prompt)


