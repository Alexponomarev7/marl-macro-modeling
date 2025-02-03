import json
import hydra
import hashlib
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import (
    Any,
    Dict,
)


def generate_hash(params: Dict) -> str:
    """
    Generate a hash from the sorted parameters.

    :param params: Dictionary of parameters
    :return: Short hash string
    """
    sorted_params = sorted(params.items())
    params_str = ','.join(f"{k}={v}" for k, v in sorted_params)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]


def generate_env_data(env, num_steps: int = 1000) -> Dict:
    """
    Generate data from the given environment using its analytical solution.

    :param env: The environment instance
    :param num_steps: Number of steps to run the environment
    :param seed: Random seed for reproducibility
    :return: A dictionary containing:
        - 'env_params': The parameters of the environment.
        - 'tracks': A DataFrame containing the generated data with columns:
            - 'state': The state of the environment at each step.
            - 'reward': The reward received at each step.
            - 'done': A boolean indicating if the episode is done.
            - 'truncated': A boolean indicating if the episode was truncated.
            - 'info': Additional information from the environment at each step.
    """
    env.reset()

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

    return {
        'env_name': env.__class__.__name__,
        'env_params': env.params,
        'action_description': env.action_description,
        'state_description': env.state_description,
        'tracks': pd.DataFrame(data),
    }

def generate_env_data_dynare(dynare_file_path: Path):
    df = pd.read_parquet(dynare_file_path)
    df["done"] = False
    return {
        "env_name": dynare_file_path.name,
        "env_params": dynare_file_path.name,
        "action_description": df.iloc[0]["action_description"],
        "state_description": df.iloc[0]["state_description"],
        "tracks": df[["state", "reward", "done", "truncated", "info"]],
    }

class DatasetWriter:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.metadata = []
        self.idx = 1

    def __enter__(self):
        return self

    def write(self, env_data: dict[str, Any], hash: str):
        output_path = self.workdir / f"{self.idx}_{hash}.parquet"
        env_data['tracks'].to_parquet(output_path)
        self.metadata.append({
            'env_name': env_data['env_name'],
            'env_params': env_data['env_params'],
            'output_dir': str(output_path),
        })


    def __exit__(self, exc_type, exc_value, traceback):
        with open(self.workdir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=4)


def run_generation_batch(dataset_cfg: dict[str, Any], envs_cfg: dict[str, Any], workdir: Path):
    """
    Run batch data generation using Hydra config for all specified environments.
    """

    metadata = []
    for env_config_metadata in dataset_cfg['envs']:
        num_steps = env_config_metadata["num_steps"]
        num_combinations = env_config_metadata["num_combinations"]
        env_config = envs_cfg[env_config_metadata["env_name"]]
        logger.info(f"Generating data for environment: {env_config['env_name']} ({num_combinations=}, {num_steps=})")

        # Generate parameter combinations for current environment
        params_list = []
        for _ in range(num_combinations):
            params = {}

            for param_name, param_spec in env_config["params"].items():
                params[param_name] = hydra.utils.instantiate(param_spec)

            params_list.append(params)

        # Run generation for each parameter combination
        logger.info(f"Generating {num_combinations} combinations")
        logger.info(f"Using {num_steps} steps per combination")

        with DatasetWriter(workdir) as writer:
            for i, params in enumerate(params_list, 1):
                logger.info(f"Running combination {i}/{num_combinations}")
                logger.info(f"Parameters: {params}")

                env = hydra.utils.instantiate({"_target_": env_config["env_class"]} | params)
                try:
                    env_data = generate_env_data(env, num_steps)
                    params_hash = generate_hash(params)
                    writer.write(env_data, params_hash)
                except Exception as e:
                    logger.error(f"Error generating data, combination {i}")
                    logger.exception(e)
                    continue

def run_generation_batch_dynare(dynare_output_path: Path, workdir: Path):
    processed_path = dynare_output_path / "processed"
    assert processed_path.exists()
    with DatasetWriter(workdir) as writer:
        for file in processed_path.glob("*.parquet"):
            env_data = generate_env_data_dynare(file)
            params_hash = generate_hash({"file_name": file.name})
            writer.write(env_data, params_hash)
