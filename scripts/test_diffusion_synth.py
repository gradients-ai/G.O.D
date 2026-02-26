#!/usr/bin/env python3
"""Test style and person synth generation using diffusion_synth functions.

Requires:
  - RUNPOD_IMAGE_SYNTH_ENDPOINT (RunPod endpoint id or full runsync URL)
  - RUNPOD_API_KEY
  - For style: .vali.env with wallet, NINETEEN_API_KEY (or signed auth) for LLM prompt gen

Usage:
  python scripts/test_diffusion_synth.py              # test both
  python scripts/test_diffusion_synth.py --person    # person only
  python scripts/test_diffusion_synth.py --style     # style only
  python scripts/test_diffusion_synth.py --num 3     # override num_prompts (default 2)
"""
import argparse
import asyncio
import os
import sys

# Load env before imports that read constants
from dotenv import load_dotenv
load_dotenv(".vali.env")

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from validator.tasks.diffusion_synth import generate_person_synthetic
from validator.tasks.diffusion_synth import generate_style_synthetic
from validator.core.config import load_config


async def test_person(num_prompts: int) -> None:
    print("\n--- Person synth ---")
    print(f"num_prompts={num_prompts}")
    image_text_pairs, ds_prefix = await generate_person_synthetic(num_prompts)
    print(f"mode={ds_prefix}, num_pairs={len(image_text_pairs)}")
    for i, p in enumerate(image_text_pairs[:5], 1):
        print(f"  {i}. image: {p.image_url[:80]}...")
        print(f"     text:  {p.text_url[:80]}...")
    if len(image_text_pairs) > 5:
        print(f"  ... and {len(image_text_pairs) - 5} more")
    print("Person synth OK\n")


async def test_style(num_prompts: int) -> None:
    print("\n--- Style synth ---")
    print(f"num_prompts={num_prompts}")
    config = load_config()
    image_text_pairs, ds_prefix = await generate_style_synthetic(config, num_prompts)
    print(f"mode={ds_prefix}, num_pairs={len(image_text_pairs)}")
    for i, p in enumerate(image_text_pairs[:5], 1):
        print(f"  {i}. image: {p.image_url[:80]}...")
        print(f"     text:  {p.text_url[:80]}...")
    if len(image_text_pairs) > 5:
        print(f"  ... and {len(image_text_pairs) - 5} more")
    print("Style synth OK\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test diffusion synth (style & person)")
    parser.add_argument("--person", action="store_true", help="Test person only")
    parser.add_argument("--style", action="store_true", help="Test style only")
    parser.add_argument("--num", type=int, default=2, help="num_prompts (default 2)")
    args = parser.parse_args()

    if not os.environ.get("RUNPOD_IMAGE_SYNTH_ENDPOINT"):
        print("Set RUNPOD_IMAGE_SYNTH_ENDPOINT (and RUNPOD_API_KEY) in env or .vali.env")
        sys.exit(1)

    run_both = not args.person and not args.style

    try:
        if run_both or args.person:
            await test_person(args.num)
        if run_both or args.style:
            await test_style(args.num)
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
