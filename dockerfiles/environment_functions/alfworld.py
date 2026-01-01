"""
This is an example of a working Rollout Function implementation for the Alfworld Environment Task.
The design of a rollout function has a huge impact on the quality of the trained model on that task.
For Environment Tasks miners are expected to implement their own rollout function.
You can always expect the environment server url for that task to be available in the env variable 'ENVIRONMENT_SERVER_URL'.
For most (if not all) tasks the environment server can be expected to have a standardized interface with /reset, /step, and /observe endpoints.
While this example is for Alfworld without the use of the AlfWorldEnvClient the design should work for all standardized environment tasks.
This is a unoptimized implementation that only trains the model on its first interaction with the environment while using a reward signal from its entire interaction.
Read more about rollout functions here: https://huggingface.co/docs/trl/main/en/openenv
"""

def alfworld_rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    from agentenv.envs import AlfWorldEnvClient
    import contextlib
    import io
    import random
    import os

    all_episode_prompt_ids: list[list[int]] = []
    all_episode_completion_ids: list[list[int]] = []
    all_episode_logprobs: list[list[float]] = []
    all_episode_rewards: list[float] = []

    # Get tokenizer once from trainer
    tokenizer = trainer.processing_class

    # Get local rank
    rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Get env server for that local rank
    raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
    server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]
    env_endpoint = server_list[rank % len(server_list)]

    # Connect to Gradients Alfworld Environment server
    DATA_LEN = 2500
    client = AlfWorldEnvClient(env_server_base=env_endpoint, data_len=DATA_LEN, timeout=2400)
    game_id = random.randint(0, DATA_LEN - 1)

    for i, prompt in enumerate(prompts):
        episode_prompt_ids: list[int] = []
        episode_completion_ids: list[int] = []
        episode_logprobs: list[float] = []
        invalid_count = 0
        done = False
        solved = False
        turn_number = 0

        # Create new environment game
        with contextlib.redirect_stdout(io.StringIO()):
            client.reset(game_id)

        # Get initial prompt for Environment Task
        with contextlib.redirect_stdout(io.StringIO()):
            initial_messages = list(client.conversation_start)

        messages = []
        # Convert format
        for message in initial_messages:
            if message["from"] == "human":
                messages.append({"role": "user", "content": message["value"]})
            elif message["from"] == "gpt":
                messages.append({"role": "assistant", "content": message["value"]})

        # Add initial state to messages
        with contextlib.redirect_stdout(io.StringIO()):
            initial_state = client.observe()
        messages.append({"role": "user", "content": initial_state})

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

            # Step through env with response from model
            with contextlib.redirect_stdout(io.StringIO()):
                step_output = client.step(completion_text)

            state, reward, done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )

            if done and reward > 0:
                solved = True

            if "Nothing happens" in state:
                invalid_count += 1

            if not done:
                with contextlib.redirect_stdout(io.StringIO()):
                    observation = client.observe()
                messages.append({"role": "user", "content": observation})

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