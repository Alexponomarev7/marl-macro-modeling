from pathlib import Path
import tempfile
import pandas as pd
from research.utils import PathStorage
from lib.dynare_traj2rl_transitions import run_model
from enum import Enum
import matplotlib.pyplot as plt
from lib.envs import NAME_TO_ENV

class GenerationType(Enum):
    DYNARE = "dynare"
    GYMNASIUM = "gymnasium"

def plot_data_file(model_name: str, sample_id: int, n_iters: int | None = None):
    config_params = PathStorage.raw_root / f"{model_name}_config_{sample_id}_config.yml"
    data_path = PathStorage.processed_root / f"{model_name}_config_{sample_id}.parquet"
    data = pd.read_parquet(data_path)

    if n_iters is not None:
        data = data.iloc[:n_iters]

    states_df = pd.DataFrame(data['state'].tolist(), columns=data['state_description'].iloc[0])
    actions_df = pd.DataFrame(data['action'].tolist(), columns=data['action_description'].iloc[0])

    combined_df = pd.concat([states_df, actions_df], axis=1)
    combined_df['reward'] = data['reward']
    combined_df['accumulated_reward'] = data['accumulated_reward']
    combined_df['truncated'] = data['truncated']
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    
    states_df.plot(ax=ax1)
    ax1.set_title('States')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    actions_df.plot(ax=ax2)
    ax2.set_title('Actions')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    combined_df['reward'].plot(ax=ax3)
    ax3.set_title('Rewards')
    
    combined_df['accumulated_reward'].plot(ax=ax4)
    ax4.set_title('Accumulated Rewards')
    
    plt.tight_layout()
    plt.show()

    with open(config_params, 'r') as f:
        print(f.read())
    return combined_df

def generate_model_dynare(model_name: str, params: dict, periods: int = 50) -> tuple[pd.DataFrame, dict]:
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_file = Path(temp_dir) / f"{model_name}.csv"

        run_model(
            input_file=Path(f"dynare/docker/dynare_models/{model_name}.mod"),
            output_file=tmp_file,
            parameters=[f"-Dperiods={periods}"] + [f"-D{param}={value}" for param, value in params.items()],
            max_retries=1,
        )

        output_file_csv = pd.read_csv(tmp_file, header=None)
        return output_file_csv, params

def generate_model_gymnasium(model_name: str, params: dict, periods: int = 50) -> tuple[pd.DataFrame, dict]:
    env = NAME_TO_ENV[model_name](**params)
    env.reset()
    states = []
    actions = []
    for _ in range(periods):
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        states.append(state)
        actions.append(action)
    df_states = pd.DataFrame(states, columns=list(env.state_description.keys()))
    df_actions = pd.DataFrame(actions, columns=list(env.action_description.keys()))
    df = pd.concat([df_states, df_actions], axis=1)
    return df, env.params

def generate_model(model_name: str, params: dict, periods: int = 50, type: GenerationType = GenerationType.DYNARE) -> tuple[pd.DataFrame, dict]:
    if type == GenerationType.DYNARE:
        return generate_model_dynare(model_name, params, periods)
    elif type == GenerationType.GYMNASIUM:
        return generate_model_gymnasium(model_name, params, periods)
    else:
        raise ValueError(f"Invalid generation type: {type}")
