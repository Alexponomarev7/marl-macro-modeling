import importlib
from typing import Callable, Optional
import pandas as pd
import numpy as np
import pickle

from loguru import logger
import hydra


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

    return reward_object


def dynare_trajectories2rl_transitions(
    input_data_path: str,
    state_columns: list[str],
    action_columns: list[str],
    reward_func: Callable,
) -> pd.DataFrame:
    """Converts Dynare trajectories into reinforcement learning transitions.

    This function reads a CSV file containing Dynare simulation data and transforms
    it into a DataFrame of reinforcement learning transitions. Each transition includes
    the current state, action, next state, reward, and additional metadata.

    Args:
        input_data_path (str): Path to the input CSV file containing Dynare data.
        state_columns (list[str]): List of column names representing the state variables.
        action_columns (list[str]): List of column names representing the action variables.
        reward_func (Callable): A function that computes the reward based on the
            current state, action, and next state.

    Returns:
        pd.DataFrame: A DataFrame containing the transitions with columns:
            - state: The current state.
            - action: The action taken.
            - next_state: The next state.
            - reward: The computed reward.
            - truncated: A boolean indicating if the episode was truncated (always False).
            - info: Additional metadata from the input data.
    """
    data = pd.read_csv(input_data_path)

    state = np.zeros(len(state_columns))
    action_columns_values = np.zeros(len(action_columns))

    transitions = []
    for idx, row in data.iterrows():
        next_state = row[state_columns].to_numpy()
        next_action_columns_values = row[action_columns].to_numpy()

        if idx == 0:
            state = next_state
            action_columns_values = next_action_columns_values
            continue

        action = next_action_columns_values - action_columns_values
        reward = reward_func(state, action, next_state)

        info_columns = list(set(row.index.to_list()) - set(state_columns + action_columns))
        info = row[info_columns].to_dict()

        transition = {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "truncated": False,
            "info": info,
        }

        state = next_state
        action_columns_values = next_action_columns_values

        transitions.append(transition)

    return pd.DataFrame(transitions)


def main() -> None:
    config_path = "../conf/dynare/"
    config_name = "config"

    with hydra.initialize(version_base=None, config_path=config_path):
        config = hydra.compose(config_name=config_name)

    for model_name, model_params in config.items():
        logger.info("Processing dynare dynamics to RL transitions...")
        transitions = dynare_trajectories2rl_transitions(
            input_data_path=model_params.input.data_path,
            state_columns=model_params.input.state_columns,
            action_columns=model_params.input.action_columns,
            reward_func=get_reward_object(model_params.reward),
        )
        logger.info("Transition successfully received.")

        logger.info("Saving data...")
        data = {
            "env_name": model_name,
            "transitions": transitions,
        }
        with open(model_params.output.data_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Data saved to {model_params.output.data_path}")


if __name__ == "__main__":
    main()


