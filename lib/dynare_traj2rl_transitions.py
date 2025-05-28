import importlib
from typing import Callable, Optional
import pandas as pd
import numpy as np
import pickle
import os
import re
from pathlib import Path

import yaml

from loguru import logger
import hydra
import pyarrow as pa
import pyarrow.parquet as pq


def get_reward_object(reward_object_path: str) -> Optional[Callable]:
    """Imports and returns a reward object from a specified path.

    This function dynamically imports a reward object (typically a class or function)
    from a given module path. The path should be in the format 'module_name.class_name'.

    Args:
        reward_object_path (str): The dotted path to the reward object,
            e.g., 'module_name.class_name'.

    Returns:
        Callable: The imported reward object (class or function). Returns None if
            the import fails due to an ImportError or AttributeError.

    """
    reward_object = None
    try:
        module_name, class_name = reward_object_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        reward_object = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Error importing reward object: {e}")

    assert reward_object is not None, f"reward object is not imported {reward_object_path=}"
    return reward_object


def dynare_trajectories2rl_transitions(
    input_data_path: str,
    state_columns: list[str],
    action_columns: list[str],
    reward_column: str,
    discount_factor: float,
) -> pd.DataFrame:
    """Converts Dynare trajectories into reinforcement learning transitions.

    Args:
        input_data_path (str): Path to the input CSV file containing Dynare data.
        state_columns (list[str]): List of column names representing the state variables.
        action_columns (list[str]): List of column names representing the action variables.
        reward_func (Callable): A function that computes the reward.
        reward_kwargs (dict): Additional keyword arguments for the reward function.

    Returns:
        pd.DataFrame: A DataFrame containing the transitions.
    """
    data = pd.read_csv(input_data_path)
    transitions = []

    current_discount_factor = 1.0
    accumulated_reward = 0.0

    for idx, row in data.iterrows():
        state = row[state_columns].to_numpy()
        action = row[action_columns].to_numpy()
        reward = float(row[reward_column])

        accumulated_reward += reward * current_discount_factor
        current_discount_factor *= discount_factor

        info_columns = list(set(row.index.to_list()) - set(state_columns + action_columns))
        info = row[info_columns].to_dict()
        info["row_id"] = idx
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "accumulated_reward": accumulated_reward,
            "truncated": False,
            "info": info,
        }

        transitions.append(transition)

    return pd.DataFrame(transitions)


def process_model_data(
    model_name: str,
    model_config: dict,
    model_params: dict,
    raw_data_path: str,
    output_dir: str,
) -> None:
    """Processes raw Dynare data for a specific model and configuration."""
    logger.info(f"Processing {model_name} with data from {raw_data_path}...")
    rl_env_conf = model_config["rl_env_settings"]

    # Extract configuration number from the filename (if any)
    config_match = re.search(r"_config_(\d+)_raw\.csv$", raw_data_path)
    config_suffix = f"_config_{config_match.group(1)}" if config_match else ""

    # Generate output path
    output_path = Path(output_dir) / f"{model_name}{config_suffix}.parquet"

    transitions = dynare_trajectories2rl_transitions(
        input_data_path=raw_data_path,
        state_columns=rl_env_conf["input"]["state_columns"],
        action_columns=rl_env_conf["input"]["action_columns"],
        reward_column=rl_env_conf["input"]["utility_column"],
        discount_factor=model_params["discount_rate"],

    )
    logger.info("Transitions successfully generated.")

    logger.info("Saving data...")

    transitions["action_description"] = pd.Series([list(rl_env_conf["input"]["action_columns"])] * len(transitions))
    transitions["state_description"] = pd.Series([list(rl_env_conf["input"]["state_columns"])] * len(transitions))

    transitions.to_parquet(output_path)

    logger.info(f"Data saved to {output_path}")

def extract_model_name(filename: str) -> str:
    """Extracts the base model name from a filename.

    Args:
        filename (str): The filename, e.g., "Born_Pfeifer_2018_MP_config_1_raw.csv".

    Returns:
        str: The base model name, e.g., "Born_Pfeifer_2018_MP".
    """
    # Удаляем суффикс "_config_*_raw.csv"
    if "_config_" in filename and "_raw" in filename:
        return filename.split("_config_")[0]
    # Если суффикса нет, возвращаем имя файла без расширения
    return filename.replace("_raw", "")


def main() -> None:
    config_path = "../dynare/conf/"
    config_name = "config"

    with hydra.initialize(version_base=None, config_path=config_path):
        config = hydra.compose(config_name=config_name)

    # Directory containing raw data files
    raw_data_dir = "./data/raw"
    output_dir = "./data/processed"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all raw data files
    raw_data_files = list(Path(raw_data_dir).glob("*_raw.csv"))

    for raw_data_file in raw_data_files:
        config_data_file = Path(str(raw_data_file).replace("_raw.csv", "_config.yml"))
        with open(config_data_file, 'r') as f:
            model_params = yaml.load(f, Loader=yaml.FullLoader)
        
        model_name = extract_model_name(raw_data_file.stem)

        process_model_data(
            model_name=model_name,
            model_config=config[model_name],
            model_params=model_params,
            raw_data_path=str(raw_data_file),
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()