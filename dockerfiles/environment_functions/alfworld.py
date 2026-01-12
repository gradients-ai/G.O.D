"""
This is an example of a working Rollout Function implementation for the Alfworld Environment Task.
The design of a rollout function has a huge impact on the quality of the trained model on that task.
For Environment Tasks miners are expected to implement their own rollout function.
You can always expect the environment server url for that task to be available in the env variable 'ENVIRONMENT_SERVER_URL'.
For most (if not all) tasks the environment server can be expected to have a standardized interface with /reset, /step, and /observe endpoints.
While this example is for Alfworld the design should work for all standardized environment tasks.
This is a unoptimized implementation that only trains the model on its first interaction with the environment while using a reward signal from its entire interaction.
Read more about rollout functions here: https://huggingface.co/docs/trl/main/en/openenv
"""

def alfworld_rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests
    import json
    
    # --- 1. Static Initialization (Once per Rank) ---
    # We check if the function has already established a connection for this worker
    if not getattr(alfworld_rollout_first_prompt_and_completion, "initialized", False):
        # Get local rank
        rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Get env server for that local rank
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]
        
        # Determine endpoint
        if not server_list:
            # Fallback (though likely fatal for the task)
            base_url = ""
            print("Warning: No ENVIRONMENT_SERVER_URLS found.")
        else:
            base_url = server_list[rank % len(server_list)]

        # Store endpoint on the function to avoid re-parsing
        alfworld_rollout_first_prompt_and_completion.base_url = base_url
        
        # Create environment (POST /create) - ONLY ONCE
        try:
            print(f"Initializing AlfWorld environment on rank {rank} at {base_url}...")
            create_res = requests.post(f"{base_url}/create", timeout=300)
            create_res.raise_for_status()
            # Store env_id on the function
            alfworld_rollout_first_prompt_and_completion.env_id = create_res.json()["id"]
            alfworld_rollout_first_prompt_and_completion.initialized = True
            print(f"Environment initialized. ID: {alfworld_rollout_first_prompt_and_completion.env_id}")
        except Exception as e:
            print(f"CRITICAL: Failed to create environment on rank {rank}: {e}")
            raise e

    # Retrieve static variables
    env_id = alfworld_rollout_first_prompt_and_completion.env_id
    env_endpoint = alfworld_rollout_first_prompt_and_completion.base_url

    # --- 2. Rollout Setup ---
    all_episode_prompt_ids: list[list[int]] = []
    all_episode_completion_ids: list[list[int]] = []
    all_episode_logprobs: list[list[float]] = []
    all_episode_rewards: list[float] = []

    tokenizer = trainer.processing_class
    DATA_LEN = 2500
    TIMEOUT = 2400

    # Hardcoded System Prompt (ReAct)
    conversation_start = [
        {
            "from": "human",
            "value": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: "THOUGHT" or "ACTION". If you choose "THOUGHT", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next action"; If you choose "ACTION", you should directly output the action in this turn. Your output must strictly follow this format:"Action:\nyour next action". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.',
        },
        {
            "from": "gpt",
            "value": "OK. I'll follow your instructions and try my best to solve the task.",
        }
    ]

    # --- 3. Batch Loop ---
    # We use a random game_id for the batch, or you could sample per item if preferred
    game_id = random.randint(0, DATA_LEN - 1)

    for i, prompt in enumerate(prompts):
        episode_prompt_ids: list[int] = []
        episode_completion_ids: list[int] = []
        episode_logprobs: list[float] = []
        invalid_count = 0
        done = False
        solved = False
        turn_number = 0
        
        # --- Reset Environment (POST /reset) ---
        # Reuse existing env_id, just change the game
        payload = {"id": env_id, "game": game_id, "world_type": "Text"}
        
        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            
            # Construct Initial Observation
            current_observation = reset_data["observation"]
            current_available_actions = reset_data["available_actions"]
            formatted_observation = f"{current_observation}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            continue

        # --- Build Conversation History ---
        messages = []
        for message in conversation_start:
            if message["from"] == "human":
                messages.append({"role": "user", "content": message["value"]})
            elif message["from"] == "gpt":
                messages.append({"role": "assistant", "content": message["value"]})
        
        messages.append({"role": "user", "content": formatted_observation})

        # --- Interaction Loop ---
        while not done and (turn_number < max_turns):
            # Generate Rollout Completion
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                episode_completion_ids = completion_ids
                episode_logprobs = logprobs

            messages.append({"role": "assistant", "content": completion_text})

            # --- Parse Action ---
            action_to_send = completion_text
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-5]

            # Parse ReAct format
            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()
            
            # --- Step Environment (POST /step) ---
            step_reward = 0.0
            step_done = False
            step_state = ""

            try:
                step_payload = {"id": env_id, "action": action_to_send}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()

                # Extract response data
                step_state = step_data["observation"]
                step_reward = step_data["reward"]
                step_done = step_data["done"]
                current_available_actions = step_data["available_actions"]
                
                # Format next observation
                formatted_observation = f"{step_state}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
                
            except Exception as e:
                print(f"Step failed: {e}")
                formatted_observation = "Invalid Action.\n\n" + formatted_observation 
                step_reward = 0.0
                step_done = False

            # Update Loop State
            if step_done and step_reward > 0:
                solved = True

            if "Nothing happens" in step_state:
                invalid_count += 1
            
            done = step_done

            if not done:
                messages.append({"role": "user", "content": formatted_observation})

            turn_number += 1
        
        train_reward = (1.0 if solved else 0.0) - 0.01 * float(invalid_count)
        all_episode_prompt_ids.append(episode_prompt_ids)
        all_episode_completion_ids.append(episode_completion_ids)
        all_episode_logprobs.append(episode_logprobs)
        all_episode_rewards.append(train_reward)

    return {
        "prompt_ids": all_episode_prompt_ids,
        "completion_ids": all_episode_completion_ids,
        "logprobs": all_episode_logprobs,
        "env_rewards": all_episode_rewards
    }

def alfworld_rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)