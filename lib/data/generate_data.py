import fire
import importlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import (
    Union,
    List,
)

from lib.config import RAW_DATA_DIR


def generate_env_data(env, num_steps: int, seed: int = 42):
    """
    Generate data from the given environment using its analytical solution.

    :param env: The environment instance
    :param num_steps: Number of steps to run the environment
    :param seed: Random seed for reproducibility
    :return: DataFrame containing the generated data
    """
    np.random.seed(seed)
    env.reset(seed=seed)

    data = []
    for _ in tqdm(range(num_steps)):
        state, reward, done, truncated, info = env.analytical_step()
        data.append({
            "state": state,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "info": info
        })

    return pd.DataFrame(data)


def main(
        env_class: str,
        num_steps: int = 1000,
        output_folder: Union[Path, str] = RAW_DATA_DIR,
        seed: int = 42,
        **env_params
):
    """
    Main function to generate data from the specified environment using the analytical solution.

    :param env_class: The environment class name (e.g., 'RBCEnv')
    :param num_steps: Number of steps to run the environment
    :param output_folder: Output folder to save the generated data to
    :param seed: Random seed for reproducibility
    :param env_params: Additional parameters to initialize the environment
    """
    logger.info(f"Importing environment {env_class} class...")
    try:
        module_name, class_name = env_class.rsplit('.', 1)
        module = importlib.import_module(module_name)
        env_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Error importing environment class: {e}")
        return

    logger.info("Initializing environment...")
    env = env_class(**env_params)

    logger.info("Generating data...")
    data = generate_env_data(env, num_steps, seed)

    logger.info("Saving data...")
    output_file = output_folder / f"{class_name}.csv"
    data.to_csv(output_file, index=False)

    logger.success(f"Data saved to {str(output_file)}")


if __name__ == '__main__':
    fire.Fire(main)
