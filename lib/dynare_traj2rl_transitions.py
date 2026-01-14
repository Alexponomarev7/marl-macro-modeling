import importlib
import shutil
import tempfile
from typing import Callable, Optional, cast
from multiprocessing import Pool, cpu_count
from collections import deque
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import pickle
import os
import re
from pathlib import Path

import yaml
import subprocess

from loguru import logger
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import traceback

from research.utils import PathStorage


class StateAccessor:
    def __init__(self, state_columns: list[str], buffer_size: int = 10):
        self.state_columns = state_columns
        self.buffer = deque(maxlen=buffer_size)
        self.raw_columns = [column[0] if isinstance(column, list) else column for column in state_columns]

    def get_columns(self) -> list[str]:
        return self.raw_columns

    def __call__(self, row: pd.Series) -> np.ndarray:
        state = []
        for column in self.state_columns:
            if isinstance(column, list):
                state_column, shift = column
                assert int(shift) < 0, "shift should be negative"
                state.append(float(self.buffer[int(shift)][state_column]))
            else:
                state.append(float(row[column]))

        self.buffer.append(pd.Series(row))
        return np.array(state)


def sample_from_range(range_values: list[float]) -> float:
    return np.random.random() * (range_values[1] - range_values[0]) + range_values[0]


# def generate_parameter_combinations(model_settings: dict, num_samples: int) -> tuple[list[list[str]], list[dict]]:
#     parameter_combinations = []
#     parameter_values = []

#     for _ in range(num_samples):
#         current_combination = []
#         current_values = {}

#         # Handle periods separately since it's not a range
#         if "periods" in model_settings:
#             current_combination.append(f"-Dperiods={model_settings['periods']}")
#             current_values["periods"] = model_settings["periods"]

#         # Sample from parameter ranges
#         if "parameter_ranges" in model_settings:
#             for param, range_values in model_settings["parameter_ranges"].items():
#                 value = sample_from_range(range_values)
#                 current_combination.append(f"-D{param}={value}")
#                 current_values[param] = value

#         if "shock_settings" in model_settings:
#             shock_settings = model_settings["shock_settings"]
#             num_shocks = shock_settings.get("num_shocks", 0)
#             current_combination.append(f"-Dnum_shocks={num_shocks}")
#             current_values["num_shocks"] = num_shocks

#             periods_available = list(range(
#                 shock_settings.get("period_range", [1, model_settings["periods"]])[0],
#                 shock_settings.get("period_range", [1, model_settings["periods"]])[1] + 1
#             ))

#             if num_shocks > 0 and len(periods_available) >= num_shocks:
#                 shock_periods = sorted(np.random.choice(periods_available, size=num_shocks, replace=False))
#             else:
#                 shock_periods = []
            
#             value_range = shock_settings.get("value_range", [-0.05, 0.05])

#             for i in range(num_shocks):
#                 if i < len(shock_periods):
#                     period = int(shock_periods[i])      
#                     value = sample_from_range(value_range)
#                 else:
#                     period = 1
#                     value = 0.0
                
#                 current_combination.append(f"-Dshock_period_{i+1}={period}")
#                 current_combination.append(f"-Dshock_value_{i+1}={value}")
#                 current_values[f"shock_period_{i+1}"] = period
#                 current_values[f"shock_value_{i+1}"] = value

#         parameter_combinations.append(current_combination)
#         parameter_values.append(current_values)

#     return parameter_combinations, parameter_values


def _generate_shock_params(
    shock_settings: dict,
    periods: int,
    prefix: str,
    shock_count_name: str,
    max_shocks: int = 6
) -> dict:
    """
    Generate shock parameters for a single shock type.
    
    Args:
        shock_settings: Settings for this shock type (num_shocks, period_range, value_range).
        periods: Total simulation periods.
        prefix: Prefix for parameter names (e.g., 'productivity_shock').
        shock_count_name: Name for the shock count parameter.
        max_shocks: Maximum number of shocks to generate.
    
    Returns:
        Dictionary with shock parameters.
    """
    params = {}
    
    num_shocks = shock_settings.get("num_shocks", 0)
    params[shock_count_name] = num_shocks
    
    period_range = shock_settings.get("period_range", [1, periods])
    period_start = max(1, period_range[0])
    period_end = min(periods, period_range[1])
    periods_available = list(range(period_start, period_end + 1))
    
    actual_num_shocks = min(num_shocks, len(periods_available))
    if actual_num_shocks > 0:
        shock_periods = sorted(np.random.choice(
            periods_available, 
            size=actual_num_shocks, 
            replace=False
        ))
    else:
        shock_periods = []
    
    value_range = shock_settings.get("value_range", [-0.05, 0.05])
    
    for i in range(max_shocks):
        if i < len(shock_periods):
            period = int(shock_periods[i])
            value = sample_from_range(value_range)
        else:
            period = 1
            value = 0.0
        
        params[f"{prefix}_period_{i+1}"] = period
        params[f"{prefix}_value_{i+1}"] = value
    
    return params


def _generate_all_shocks(
    shocks_config: dict,
    periods: int,
    max_shocks_per_type: int = 6
) -> dict:
    """
    Generate parameters for all shock types.
    
    Args:
        shocks_config: Dictionary where keys are shock names and values are settings.
        periods: Total simulation periods.
        max_shocks_per_type: Maximum number of shocks per type.
    
    Returns:
        Dictionary with all shock parameters for Dynare.
    """
    all_params = {}
    
    for shock_name, shock_settings in shocks_config.items():
        shock_params = _generate_shock_params(
            shock_settings=shock_settings,
            periods=periods,
            prefix=f"{shock_name}_shock",
            shock_count_name=f"num_{shock_name}_shocks",
            max_shocks=max_shocks_per_type
        )
        all_params.update(shock_params)
    
    return all_params


def generate_parameter_combinations(model_settings: dict, num_samples: int) -> tuple[list[list[str]], list[dict]]:
    parameter_combinations = []
    parameter_values = []

    for _ in range(num_samples):
        current_combination = []
        current_values = {}

        # Handle periods separately since it's not a range
        if "periods" in model_settings:
            current_combination.append(f"-Dperiods={model_settings['periods']}")
            current_values["periods"] = model_settings["periods"]

        # Sample from parameter ranges
        if "parameter_ranges" in model_settings:
            for param, range_values in model_settings["parameter_ranges"].items():
                value = sample_from_range(range_values)
                current_combination.append(f"-D{param}={value}")
                current_values[param] = value

        if "shocks" in model_settings:
            shock_params = _generate_all_shocks(
                shocks_config=model_settings["shocks"],
                periods=model_settings.get("periods", 100)
            )
            for key, value in shock_params.items():
                current_combination.append(f"-D{key}={value}")
                current_values[key] = value

        parameter_combinations.append(current_combination)
        parameter_values.append(current_values)

    return parameter_combinations, parameter_values


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


def run_model(
    input_file: Path, 
    output_file: Path,
    output_params_file: Path,
    parameters: list[str],
    max_retries: int = 3,
) -> None:
    """Run a Dynare model with specified parameters and save results.
    
    Args:
        input_file: Path to the Dynare .mod file
        output_file: Path to save the output CSV
        periods: Number of simulation periods
        parameters: List of parameter strings to pass to Dynare
        max_retries: Maximum number of retry attempts
    """
    retries = 0
    while retries < max_retries:
        print(f"Running model: {input_file} (attempt {retries + 1})")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                input_tmp_file = Path(tmp_dir) / input_file.name
                shutil.copy(input_file, input_tmp_file)

                # Run Dynare model
                cmd: list[str] = [
                    "octave",
                    "--eval",
                    f"""
                    addpath {os.environ["DYNARE_PATH"]}; 
                    cd {input_tmp_file.parent}; 
                    dynare {input_tmp_file.name} {' '.join(parameters)}; 
                    oo_simul = oo_.endo_simul'; 
                    var_names = M_.endo_names_tex;
                    param_names = M_.param_names;
                    param_values = M_.params;

                    fid = fopen('{output_file}', 'w');
                    fprintf(fid, '%s,', var_names{{1:end-1}});
                    fprintf(fid, '%s\\n', var_names{{end}});
                    fclose(fid);

                    dlmwrite('{output_file}', oo_simul, '-append');
                    fid = fopen('{output_params_file}', 'w');
                    for i = 1:length(param_names)
                        fprintf(fid, '%s: %f\\n', char(param_names(i)), param_values(i));
                    end
                    fclose(fid);
                    """
                ]   

                process = subprocess.run(cmd, capture_output=True, text=True)
            logger.info(f"Running command: {subprocess.list2cmdline(cmd)}")

            if process.returncode != 0:
                raise RuntimeError(f"Dynare failed with error: {process.stdout} {process.stderr}")
            
            print(f"Model {input_file} completed successfully.")
            return

        except Exception as e:
            print(f"Error running model {input_file}:")
            print(f"Error message: {str(e)}")
            print("Stack trace:")
            traceback.print_exc()
            
            retries += 1
            if retries < max_retries:
                print(f"Retrying model {input_file}...")
            else:
                print(f"Failed to run model {input_file} after {max_retries} attempts.")
    
    if retries == max_retries:
        raise RuntimeError(f"Failed to run model {input_file} after {max_retries} attempts.")


def process_model_combination(args):
    _, input_file, base_name, combination, values, raw_data_dir = args
    output_file = os.path.join(os.getcwd(), raw_data_dir, base_name + "_raw.csv")
    output_params_file = os.path.join(os.getcwd(), raw_data_dir, base_name + "_params.yaml")
    config_file = os.path.join(os.getcwd(), raw_data_dir, base_name + "_config.yml")
    
    run_model(Path(input_file), Path(output_file), Path(output_params_file), combination)
    print(f"Output saved to {output_file}")
    print(f"Config saved to {config_file}")
    print(f"Params saved to {output_params_file}")


def run_models(config: dict, raw_data_dir: Path) -> list[tuple[Path, Path]]:
    output_files = []
    tasks = []

    for model_name, model_config in config.items():
        model_settings = model_config["dynare_model_settings"]
        num_samples = model_settings["num_samples"]
        parameter_combinations, parameter_values = generate_parameter_combinations(model_settings, num_samples)
        
        for i, (combination, values) in enumerate(zip(parameter_combinations, parameter_values)):
            input_file = os.path.join(PathStorage().dynare_configs_root, model_name + ".mod")
            base_name = "_".join([model_name, f"config_{i}"])
            output_file = os.path.join(os.getcwd(), raw_data_dir, base_name + "_raw.csv")
            output_params_file = os.path.join(os.getcwd(), raw_data_dir, base_name + "_params.yaml")
            output_files.append((Path(output_file), Path(output_params_file)))
            task = (model_name, input_file, base_name, combination, values, raw_data_dir)
            tasks.append(task)
    
    num_processes = min(min(cpu_count(), len(tasks)), 32)
    print(f"Running {len(tasks)} tasks using {num_processes} processes")
    
    with Pool(processes=num_processes) as pool:
        pool.map(process_model_combination, tasks)

    return output_files


def dynare_trajectories2rl_transitions(
    input_data_path: Path,
    state_accessor: StateAccessor,
    endogenous_accessor: StateAccessor,
    action_columns: list[str],
    reward_fn: Callable,
    reward_kwargs: dict,
    discount_factor: float,
    model_params: dict,
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

    data["REWARD_COMPUTED"] = reward_fn(data, model_params, **reward_kwargs)

    for idx, row in data.iterrows():
        if idx == 0:
            # first row is the initial state
            state_accessor.buffer.append(row)
            endogenous_accessor.buffer.append(row)
            continue

        state = state_accessor(row)
        endogenous = endogenous_accessor(row)
        action = row[action_columns].to_numpy()
        reward = float(row["REWARD_COMPUTED"])

        accumulated_reward += reward * current_discount_factor
        current_discount_factor *= discount_factor

        info_columns = list(set(row.index.to_list()) - set(state_accessor.get_columns() + action_columns))
        row_info = row[info_columns].to_dict()
        row_info["row_id"] = idx
        row_info["model_params"] = model_params
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "endogenous": endogenous,
            "accumulated_reward": accumulated_reward,
            "truncated": False,
            "info": row_info,
        }

        transitions.append(transition)
    return pd.DataFrame(transitions)


def process_model_data(
    model_name: str,
    model_config: dict,
    model_params: dict,
    raw_data_path: Path,
    output_dir: Path,
) -> None:
    """Processes raw Dynare data for a specific model and configuration."""
    logger.info(f"Processing {model_name} with data from {raw_data_path}...")
    rl_env_conf = model_config["rl_env_settings"]

    # Extract configuration number from the filename (if any)
    config_match = re.search(r"_config_(\d+)_raw\.csv$", str(raw_data_path))
    config_suffix = f"_config_{config_match.group(1)}" if config_match else ""

    # Generate output path
    output_path = Path(output_dir) / f"{model_name}{config_suffix}.parquet"

    state_accessor = StateAccessor(rl_env_conf["input"]["state_columns"])
    endogenous_accessor = StateAccessor(rl_env_conf["input"]["endogenous_columns"])

    reward_fn = get_reward_object(rl_env_conf["reward"])
    transitions = dynare_trajectories2rl_transitions(
        input_data_path=raw_data_path,
        state_accessor=state_accessor,
        endogenous_accessor=endogenous_accessor,
        action_columns=rl_env_conf["input"]["action_columns"],
        reward_fn=reward_fn, # type: ignore
        reward_kwargs=rl_env_conf["reward_kwargs"],
        discount_factor=model_params["beta"],
        model_params=model_params,
    )
    logger.info("Transitions successfully generated.")

    logger.info("Saving data...")

    transitions["action_description"] = pd.Series([list(rl_env_conf["input"]["action_columns"])] * len(transitions))
    transitions["state_description"] = pd.Series([list(state_accessor.get_columns())] * len(transitions))
    transitions["endogenous_description"] = pd.Series([list(endogenous_accessor.get_columns())] * len(transitions))

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


@hydra.main(config_path="../dynare/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    config = cast(dict, OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    path_storage = PathStorage(data_folder=config["metadata"]["data_folder"])
    path_storage.raw_root.mkdir(parents=True, exist_ok=True)
    path_storage.processed_root.mkdir(parents=True, exist_ok=True)
    logger.info("Running models...")
    output_files = run_models(config["models"], path_storage.raw_root)
    logger.info("Models run successfully.")

    # Ensure output directory exists
    os.makedirs(path_storage.raw_root, exist_ok=True)
    os.makedirs(path_storage.processed_root, exist_ok=True)
    for raw_data_file, params_file in output_files:
        with open(params_file, 'r') as f:
            model_params = yaml.load(f, Loader=yaml.FullLoader)
        
        model_name = extract_model_name(raw_data_file.stem)

        if model_name not in config["models"]:
            logger.warning(f"Model {model_name} not found in config")
            continue

        try:
            process_model_data(
                model_name=model_name,
                model_config=config["models"][model_name],
                model_params=model_params,
                raw_data_path=raw_data_file,
                output_dir=path_storage.processed_root,
            )
        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}")
            logger.error(traceback.format_exc())
            continue


if __name__ == "__main__":
    main()
