import os

import toml
import yaml

import trainer.constants as train_cst


def update_flash_attention(config: dict, model: str) -> dict:
    # You might want to make this model-dependent.
    config["flash_attention"] = False
    return config


def save_config(config: dict, config_path: str) -> None:
    with open(config_path, "w") as file:
        yaml.dump(config, file)


def save_config_toml(config: dict, config_path: str) -> None:
    with open(config_path, "w") as file:
        toml.dump(config, file)


def create_reward_funcs_file(
    reward_funcs: list[str],
    task_id: str,
    destination_dir: str = train_cst.AXOLOTL_DIRECTORIES["src"],
) -> tuple[str, list[str]]:
    """
    Create a Python file with reward functions for GRPO training.

    Args:
        reward_funcs: Python reward function implementations.
        task_id: Unique task identifier.
        destination_dir: Directory where Axolotl can import the generated module.
    """
    filename = f"rewards_{task_id}"
    filepath = os.path.join(destination_dir, f"{filename}.py")

    func_names = []
    for reward_func in reward_funcs:
        if "def " in reward_func:
            func_name = reward_func.split("def ")[1].split("(")[0].strip()
            func_names.append(func_name)

    with open(filepath, "w") as f:
        f.write("# Auto-generated reward functions file\n\n")
        for reward_func in reward_funcs:
            f.write(f"{reward_func}\n\n")

    return filename, func_names
