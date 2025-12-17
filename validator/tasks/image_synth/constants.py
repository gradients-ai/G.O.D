import os


FACE_IMAGE_PATH = "validator/tasks/image_synth/ComfyUI/input/person.jpg"
FACE_IMAGE_URL = "https://thispersondoesnotexist.com/"
LLAVA_MODEL_PATH = "/app/validator/tasks/image_synth/cache/llava-v1.5-7b"
PERSON_WORKFLOW_PATH = "validator/tasks/image_synth/person_avatars_template.json"
STYLE_WORKFLOW_PATH = "validator/tasks/image_synth/style_template.json"
DEFAULT_SAVE_DIR = "/app/images/"
PROMPT_EXAMPLES = """
    1. portrait photo of Daniel, a stylish man in his 30s, walking through a neon-lit street at night, wearing a black leather jacket and jeans, glowing signs reflecting on puddles, cinematic lighting, shallow depth of field, realistic, 85mm lens
    2. photo of Ava, a young woman with auburn hair and freckles, sitting on a wooden fence in a peaceful countryside, wearing a flowy white dress, tall grass and wildflowers around her, golden hour sunlight, soft focus, realistic, DSLR look
    3. photo of Malik, a Black man in his late 20s, holding a transparent umbrella on a rainy street, wearing a navy trench coat and scarf, soft raindrops visible, urban background blurred, cinematic tones, shallow depth of field, realistic
    4. realistic photo of Lena, a fit woman in her 20s with blonde beach waves, wearing a sporty bikini, jogging along the shoreline, waves crashing behind her, bright sunlight and blue sky, captured with a fast shutter, 50mm lens
    5. portrait of Takeshi, an Asian man in his 40s with short black hair, sitting in a neon-lit futuristic cyberpunk-style cafe, wearing a smart-casual blazer, glowing menus in the background, soft reflections on glass table, realistic lighting, bokeh
    """


# prompt stuff
NUM_PROMPTS = int(os.getenv("NUM_PROMPTS", 15))
PERSON_PROMPT = f"""
        Here is an image of a person named 'person_name'. Generate {NUM_PROMPTS} different prompts for creating an avatar of the person - make sure their name is listed in the prompt.
        Place them in different places, backgrounds, scenarios, and emotions.
        Use different settings like beach, house, room, park, office, city, and others.
        Also use a different range of emotions like happy, sad, smiling, laughing, angry, thinking for every prompt.
        Here are a few examples of the prompts to get you started, getting inspiration from these, you can try to create more for 'person_name': 
        {PROMPT_EXAMPLES}
        """
