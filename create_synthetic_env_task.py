#!/usr/bin/env python3
"""Test script to create a synthetic environment task standalone."""
import asyncio
import os
import sys
import traceback
from validator.core.config import load_config
from validator.tasks.synthetic_scheduler import (
    _get_text_models,
    _get_instruct_text_datasets,
    create_synthetic_env_task,
)
from validator.utils.util import try_db_connections


async def test_db_connection(config):
    """Test database connection before creating task."""
    try:
        print("Testing database connections...")
        await try_db_connections(config)
        print("✓ Database connections successful")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        traceback.print_exc()
        return False


async def create_env_task():
    """Create a synthetic environment task."""
    print("Loading configuration...")
    config = load_config()
    
    # Test DB connection first
    print("\nTesting database connection...")
    if not await test_db_connection(config):
        print("ERROR: Database connection failed. Please check your database configuration.")
        sys.exit(1)
    
    try:
        print("\nGetting text models...")
        models = _get_text_models(config.keypair)
        
        # Environment tasks don't need real datasets - function uses dummy dataset
        # Still need to pass datasets parameter for function signature compatibility
        datasets = _get_instruct_text_datasets(config.keypair)
        
        print("\nCreating synthetic environment task...")
        task = await create_synthetic_env_task(config, models, datasets)
        
        sys.stdout.flush()
        
        print(f"\n✓ Successfully created environment task!", flush=True)
        print(f"  Task ID: {task.task_id}", flush=True)
        print(f"  Dataset ID: {task.ds}", flush=True)
        print(f"  Model ID: {task.model_id}", flush=True)
        print(f"  Environment: {task.environment_name}", flush=True)
        print(f"  Status: {task.status}", flush=True)
        print(f"  Hours to complete: {task.hours_to_complete}", flush=True)
        print(f"  Created at: {task.created_at}", flush=True)
        print(f"  Termination at: {task.termination_at}", flush=True)
        print(f"  Account ID: {task.account_id}", flush=True)
        print(f"  Yarn factor: {task.yarn_factor}", flush=True)
        
        # Exit immediately from here to avoid asyncio.run() cleanup hanging
        print("\n✓ Task creation completed successfully!", flush=True)
        os._exit(0)
        
    except Exception as e:
        print(f"\n✗ Failed to create environment task: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        asyncio.run(create_env_task())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        os._exit(1)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        traceback.print_exc()
        os._exit(1)

