import pandas as pd
from research.utils import PathStorage
import matplotlib.pyplot as plt

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

def generate_model_with_custom_params(model_name: str, params: dict):
    pass