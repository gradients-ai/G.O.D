import json
import os
import uuid
from copy import deepcopy

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


