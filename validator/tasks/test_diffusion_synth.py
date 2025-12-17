import asyncio
from typing import AsyncGenerator

from core.models.payload_models import ImageModelInfo
from core.models.utility_models import ImageModelType
from validator.core.config import load_config
from validator.tasks.diffusion_synth import create_synthetic_image_task


async def mock_image_models_generator() -> AsyncGenerator[ImageModelInfo, None]:
    """Mock generator that yields a test image model."""
    # Yield a test model - you can change this to match a real model if needed
    yield ImageModelInfo(
        model_id="test-model-id",
        model_type=ImageModelType.SDXL,  # Change to ImageModelType.FLUX if you want to test FLUX
    )


async def test():
    print("=" * 80)
    print("Testing create_synthetic_image_task (FULL TEST)")
    print("=" * 80)
    print("\nLoading config...")
    config = load_config()
    
    print("Creating synthetic image task...")
    try:
        task = await create_synthetic_image_task(config, mock_image_models_generator())
        print(f"\n✅ Task created successfully!")
        print(f"Task ID: {task.task_id}")
        print(f"Dataset prefix: {task.ds}")
        print(f"Model ID: {task.model_id}")
        print(f"Number of image-text pairs: {len(task.image_text_pairs) if hasattr(task, 'image_text_pairs') else 'N/A'}")
        if hasattr(task, 'image_text_pairs'):
            print("\nImage-Text Pair URLs:")
            for i, pair in enumerate(task.image_text_pairs):
                print(f"  Pair {i+1}:")
                print(f"    Image: {pair.image_url}")
                print(f"    Text:  {pair.text_url}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test())

