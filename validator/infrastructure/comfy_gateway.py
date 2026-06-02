import io
import json
import time
import urllib.parse
import urllib.request
import uuid

import websocket
from PIL import Image


# Configuration variables
server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())
ws = None  # WebSocket connection object


def connect():
    global ws
    ws = websocket.WebSocket()
    while True:
        try:
            ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
            print("Connected to WebSocket.")
            break
        except ConnectionRefusedError:
            print("Could not connect to ComfyUI because it is not up yet. Sleeping for 2 seconds before trying again.")
            time.sleep(2)


def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
        return response.read()


def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())


def get_images(prompt):
    prompt_id = queue_prompt(prompt)["prompt_id"]
    output_images = {}

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    break  # Execution is done
        else:
            continue  # previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history["outputs"]:
        for node_id in history["outputs"]:
            node_output = history["outputs"][node_id]
            if "images" in node_output:
                images_output = []
                for image in node_output["images"]:
                    image_data = get_image(image["filename"], image["subfolder"], image["type"])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images


def generate(payload):
    img_list = []
    images = get_images(payload)
    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
            img_list.append(image)

    return img_list
