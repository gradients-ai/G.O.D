#!/usr/bin/env python3
"""
Script to add the new affine reward functions to the database via API endpoint.
"""
import asyncio
import inspect
import os

import asyncpg
import requests

from validator.core import constants as cst
from validator.utils.affine_reward_functions import abd_reward_function
from validator.utils.affine_reward_functions import ded_reward_function
from validator.utils.affine_reward_functions import sat_reward_function


def load_env_file():
    try:
        with open('.vali.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key in ['DATABASE_URL', 'FRONTEND_API_KEY', 'VALIDATOR_PORT']:
                        value = value.strip().strip('"').strip("'")

                        os.environ[key] = value
                        print(f"🔧 Loaded {key} from .vali.env")
    except FileNotFoundError:
        print("⚠️  .vali.env file not found")
        return None

async def delete_existing_functions(connection_string):
    pool = await asyncpg.create_pool(connection_string)
    
    try:
        async with pool.acquire() as conn:
            function_names = ['sat_reward_function', 'abd_reward_function', 'ded_reward_function']
            for func_name in function_names:
                query = """
                    DELETE FROM reward_functions 
                    WHERE reward_func LIKE $1
                """
                pattern = f"%def {func_name}%"
                result = await conn.execute(query, pattern)
                print(f"🗑️  Deleted existing {func_name} (if it existed)")
    finally:
        await pool.close()

async def main():
    load_env_file()
    
    connection_string = os.getenv("DATABASE_URL")
    
    if not connection_string:
        print("❌ ERROR: DATABASE_URL not found in environment or .vali.env file")
        return
    
    print("🚀 Setting up affine reward functions via API endpoint...")
    print(f"Database URL: {connection_string.split('@')[1] if '@' in connection_string else 'localhost'}")
    
    print("\n🗑️  Cleaning up existing functions...")
    await delete_existing_functions(connection_string)
    
    reward_functions = [
        ("SAT Reward Function", "Partial credit reward function for SAT problems", sat_reward_function),
        ("ABD Reward Function", "Partial credit reward function for ABD problems", abd_reward_function),
        ("DED Reward Function", "Partial credit reward function for DED problems", ded_reward_function),
    ]
    
    print(f"\n📋 Functions to add via API:")
    for i, (name, desc, func) in enumerate(reward_functions, 1):
        print(f"  {i}. {name} ({func.__name__})")
    
    validator_port = os.getenv("VALIDATOR_PORT", "9001")
    api_key = os.getenv("FRONTEND_API_KEY")
    api_base = f"http://localhost:{validator_port}"
    endpoint = f"{api_base}/v1/grpo/reward_functions"
    
    print(f"\n🌐 Using API endpoint: {endpoint}")
    if api_key:
        print(f"🔐 Using API key: {api_key[:8]}...")
    else:
        print("⚠️  No FRONTEND_API_KEY found in environment")
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    added_ids = []
    for name, description, func in reward_functions:
        func_code = inspect.getsource(func)
        
        payload = {
            "name": name,
            "description": description,
            "code": func_code,
            "weight": 1.0
        }
        
        print(f"\n📤 Posting {func.__name__}...")
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                reward_id = result.get("reward_id")
                if reward_id:
                    added_ids.append(reward_id)
                    print(f"✅ Successfully added {func.__name__}")
                    print(f"   ID: {reward_id}")
                else:
                    print(f"❌ API response missing reward_id: {result}")
            else:
                print(f"❌ API request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error posting {func.__name__}: {e}")
    
    if added_ids:
        print(f"\n🔍 Verifying functions were added to database...")
        
        pool = await asyncpg.create_pool(connection_string)
        
        try:
            async with pool.acquire() as conn:
                for name, description, func in reward_functions:
                    query = f"""
                        SELECT reward_id, func_hash 
                        FROM reward_functions 
                        WHERE reward_func LIKE $1
                        ORDER BY created_at DESC
                        LIMIT 1
                    """
                    pattern = f"%def {func.__name__}%"
                    result = await conn.fetchrow(query, pattern)
                    
                    if result:
                        reward_id = result['reward_id']
                        func_hash = result['func_hash'][:12]
                        print(f"✅ {func.__name__}:")
                        print(f"   ID: {reward_id}")
                        print(f"   Hash: {func_hash}...")
                    else:
                        print(f"❌ {func.__name__}: Not found in database")
                        
        finally:
            await pool.close()
    else:
        print(f"\n❌ No functions were successfully added via API")
    
    if len(added_ids) == 3:
        print(f"\n🔄 Auto-updating constants file...")
        
        constants_path = "validator/core/constants.py"
        with open(constants_path, 'r') as f:
            content = f.read()
        
        old_pattern = r'AFFINE_REWARD_FN_IDS = \[[\s\S]*?\]'
        new_ids_str = f'''AFFINE_REWARD_FN_IDS = [
    "{added_ids[0]}",  # sat_reward_function
    "{added_ids[1]}",  # abd_reward_function  
    "{added_ids[2]}",  # ded_reward_function
]'''
        
        import re
        updated_content = re.sub(old_pattern, new_ids_str, content)
        
        with open(constants_path, 'w') as f:
            f.write(updated_content)
            
        print(f"✅ Updated {constants_path} with actual reward function IDs")
        print(f"\n📝 New constants:")
        print(f"AFFINE_REWARD_FN_IDS = {added_ids}")
        
        print(f"\n⚠️  NEXT STEPS:")
        print(f"1. Restart your validator to pick up the new constants")
        print(f"2. Check validator logs for 'Looking for affine reward functions' messages")
        print(f"3. Verify all 3 functions are found and loaded")
    else:
        print(f"\n❌ ERROR: Expected 3 function IDs, got {len(added_ids)}")
        print(f"📝 Current constants in validator/core/constants.py:")
        print(f"AFFINE_REWARD_FN_IDS = {cst.AFFINE_REWARD_FN_IDS}")
        print(f"\n⚠️  MANUAL STEPS REQUIRED:")
        print(f"1. Check API responses above for any errors")
        print(f"2. Manually update AFFINE_REWARD_FN_IDS if needed")
        print(f"3. Restart your validator to pick up the new constants")


if __name__ == "__main__":
    asyncio.run(main())