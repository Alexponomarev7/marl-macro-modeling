import json
import hydra
import hashlib
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import (
    Any,
    Dict,
)

from research.utils import PathStorage


def generate_hash(params: Dict) -> str:
    """
    Generate a hash from the sorted parameters.

    :param params: Dictionary of parameters
    :return: Short hash string
    """
    sorted_params = sorted(params.items())
    params_str = ','.join(f"{k}={v}" for k, v in sorted_params)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]


def _to_scalar(x: Any) -> float:
    """Extract a python float from either a raw number or a shape-(1,) array (both
    conventions are used across lib/envs/*.py state dicts)."""
    return float(np.asarray(x, dtype=np.float64).reshape(-1)[0])


def generate_env_data(env, num_steps: int = 1000) -> Dict:
    """
    Generate data from the given environment using its analytical solution.

    :param env: The environment instance
    :param num_steps: Number of steps to run the environment
    :param seed: Random seed for reproducibility
    :return: A dictionary containing:
        - 'env_params': The parameters of the environment.
        - 'tracks': A DataFrame with columns matching the processed Dynare episodes,
          so both are consumable by lib.dataset.EconomicsDataset: 'state', 'action', 'endogenous',
          'reward', 'done', 'truncated', 'info' (with state_description/action_description/
          endogenous_description embedded in 'info', matching each row).
    """
    env.reset()

    state_description = env.state_description
    action_description = env.action_description
    state_names = list(state_description.keys())
    action_names = list(action_description.keys())
    # EconomicsDataset needs a flat numeric model_params dict; env.params may also carry
    # non-numeric entries (e.g. utility_function name/params), which don't belong there.
    model_params = {k: v for k, v in env.params.items() if isinstance(v, (int, float))}

    rows = []
    for _ in tqdm(range(num_steps)):
        state, reward, done, truncated, info = env.analytical_step()
        if "action" not in info:
            raise KeyError(
                f"{env.__class__.__name__}.analytical_step() info is missing 'action'; "
                "cannot build training data without the action taken at each step."
            )
        row_info = dict(info)
        row_info.update({
            "state_description": state_names,
            "action_description": action_names,
            "endogenous_description": [],
            "model_params": model_params,
        })
        rows.append({
            "state": np.array([_to_scalar(state[name]) for name in state_names], dtype=np.float32),
            "action": np.array([_to_scalar(v) for v in info["action"]], dtype=np.float32),
            "endogenous": np.array([], dtype=np.float32),
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "info": row_info,
        })

    return {
        'env_name': env.__class__.__name__,
        'env_group': env.__class__.__name__,
        'env_params': env.params,
        'action_description': action_description,
        'state_description': state_description,
        'tracks': pd.DataFrame(rows),
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
        env_group = env_data["env_group"]
        self.metadata.append({
            'env_name': env_data['env_name'],
            'env_group': env_group,
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

def run_generation_batch_dynare(
    dynare_output_path: Path,
    workdir: Path,
    include_models: list[str] | None = None,
    exclude_models: list[str] | None = None,
):
    """Index the processed Dynare episodes in place (no copies) for EconomicsDataset.

    include_models / exclude_models: Dynare model names to keep / drop.
    """
    processed_path = dynare_output_path

    assert processed_path.exists(), f"processed path {processed_path} does not exist"
    files = sorted(processed_path.glob("*.parquet"))
    models = {f: f.name.rsplit("_config_", 1)[0] for f in files}
    for name, selection in (("include_models", include_models), ("exclude_models", exclude_models)):
        unknown = sorted(set(selection or []) - set(models.values()))
        if unknown:
            raise KeyError(f"{name} lists models with no episodes in {processed_path}: {unknown}")
    if include_models:
        files = [f for f in files if models[f] in include_models]
    if exclude_models:
        files = [f for f in files if models[f] not in exclude_models]
    metadata = []
    for file in files:
        parquet = pq.ParquetFile(file)
        if parquet.metadata.num_rows == 0:
            logger.warning(f"Skipping empty dynare episode parquet: {file}")
            continue
        info = next(parquet.iter_batches(batch_size=1, columns=["info"])).to_pylist()[0]["info"]
        metadata.append({
            "env_name": file.name,
            "env_group": info["env_group"],
            "env_params": file.name,
            "output_dir": str(file.resolve()),
        })
    with open(workdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

class DatasetGenerator:
    """Handles the creation and organization of datasets."""

    def __init__(self, dataset_cfg: dict[str, Any]):
        """
        Initialize DatasetCreator with configuration.
        Args:
            dataset_cfg: Configuration dictionary for dataset creation
        """
        self.cfg = dataset_cfg
        self.workdir = Path(dataset_cfg['workdir'])
        self.enabled = dataset_cfg['enabled']

    # todo: rm
    def create(self):
        """Generate datasets for all stages (train, val, test)."""
        if not self.enabled:
            logger.info("dataset generation is disabled, skipping")
            return

        logger.info("stage 1: data generation")
        self.workdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"WorkDir: {self.workdir}")

        for stage in ['train', 'val']:
            logger.info(f"generating stage: {stage}")
            stage_dir = self.workdir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)

            stage_cfg = self.cfg[stage]
            if stage_cfg['type'] == 'envs':
                run_generation_batch(stage_cfg, self.cfg['envs'], stage_dir)
            elif stage_cfg['type'] == 'dynare':
                run_generation_batch_dynare(
                    PathStorage(stage_cfg['dynare_output_path']).processed_root,
                    stage_dir,
                    include_models=stage_cfg.get('include_models'),
                    exclude_models=stage_cfg.get('exclude_models'),
                )
            else:
                raise ValueError(f"Unknown dataset type: {stage_cfg['type']}")


@hydra.main(config_name='default.yaml', config_path="../pipeline/configs/dataset", version_base=None)
def main(cfg: DictConfig):
    dataset_generator = DatasetGenerator(cfg)
    dataset_generator.create()

if __name__ == "__main__":
    main()
