import json
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from loguru import logger as log

STATE_MAPPING = {
    "Empty": 0,
    "Output": 1,
    "Consumption": 2,
    "Capital": 3,
    "LoggedProductivity": 4,
    "Debt": 5,
    "InterestRate": 6,
    "PreferenceShock": 7,
    "CountryPremiumShock": 8,
    "TechGrowthRate": 9,
    "MUConsumption": 10,
    # "Hours Worked": 4,
    # "Total Factor Productivity": 5,
    # "Annualized Interest Rate": 6,
    # "Real Wage": 7,
    # "Investment": 8,
    # "Technology Shock": 9,
    # "Labor": 10,
    # "Preference Shock": 11,
    # "Marginal Utility": 12,
    # "Utility": 13,
    # "Price Inflation": 14,
    # "Debt": 15,
    # "Trade Balance To Output Ratio": 16,
    # "Trade Balance to Output Ratio": 16,
    # "Output Gap": 17,
    # "Current Account To Output Ratio": 18,
    # "Natural Output": 19,
    # "Output Deviation From Steady State": 20,
    # "Bond Price": 21,
    # "Government Spending": 22,
    # "Marginal Utility of Consumption": 23,
    # "Marginal Utility of Labor": 24,
    # "Consumption to GDP Ratio": 25,
    # "Investment to GDP Ratio": 26,
    # "Net Exports": 27,
    # "Log Output": 28,
    # "Log Consumption": 29,
    # "Log Investment": 30,
    # "Output Growth": 31,
    # "Natural Interest Rate": 32,
    # "Real Interest Rate": 33,
    # "Nominal Interest Rate": 34,
    # "Real Money Stock": 35,
    # "Money Growth Annualized": 36,
    # "Nominal Money Stock": 37,
    # "AR(1) Monetary Policy Shock Process": 38,
    # "AR(1) Technology Shock Process": 39,
    # "AR(1) Preference Shock Process": 40,
    # "Price Level": 41,
    # "Nominal Wage": 42,
    # "Real Wage Gap": 43,
    # "Wage Inflation": 44,
    # "Natural Real Wage": 45,
    # "Markup": 46,
    # "Annualized Wage Inflation Rate": 47,
    # "Value Function": 48,
    # "Auxiliary Variable For Value Function": 49,
    # "Expected Stochastic Discount Factor": 50,
    # "Volatility": 51,
    # "Expected Return On Capital": 52,
    # "Risk-Free Rate": 53,
    # "Money Growth": 54,
    # "Output Growth Rate": 55,
    # "Consumption Growth Rate": 56,
    # "Investment Growth Rate": 57,
    # "Technology Growth Rate": 58,
    # "Interest Rate": 59,
    # "Country Premium Shock": 60,
    # "Productivity": 61,
    # "Real Return On Capital": 62,
    # "Real Consumption": 63,
    # "Money Stock": 64,
    # "Growth Rate Of Money Stock": 65,
    # "Foreign Price Level": 66,
    # "Foreign Bonds": 67,
    # "Foreign Interest Rate": 68,
    # "Exchange Rate": 69,
    # "Log Capital Stock": 70,
    # "Log Labor": 71,
    # "Log Real Wage": 72,
    # "Annualized Real Interest Rate": 73,
    # "Annualized Nominal Interest Rate": 74,
    # "Annualized Natural Interest Rate": 75,
    # "Annualized Inflation Rate": 76,
    # "Trade Balance": 77,
    # "Capital Stock": 78,
    # "Real Output": 79,
    # "Output Minus Consumption": 80,
    # "Lagrange Multiplier A": 81,
    # "Lagrange Multiplier B": 82,
    # "Inflation Rate": 83,
    # "Inflation": 83,
    # "Marginal Costs": 84,
    # "Market Tightness": 85,
    # "Log TFP": 86,
    # "Log Vacancies": 87,
    # "Log Wages": 88,
    # "Log Unemployment": 89,
    # "Log Tightness A": 90,
    # "Log Tightness B": 91,
    # "Vacancies": 92,
    # "Unemployment Rate": 93,
    # "Matches": 94,
    # "Meeting Rate Between Firms And Workers": 95,
    # "Employment": 96,
    # "Gross Output A": 97,
    # "Gross Output B": 98,
    # "Government Spending Shock": 99,
    # "AR(1) Technology Process": 100,
}

ACTION_MAPPING = {
    "Empty": 0,
    "Investment": 1,
    "Consumption": 2,
    "HoursWorked": 3,
}

ENV_MAPPING = {
    "Born_Pfeifer_2018_MP": 0,
    "Aguiar_Gopinath_2007": 1,
    "RBC_news_shock_model": 2,
    "Hansen_1985": 3,
    "GarciaCicco_et_al_2010": 4,
    "Caldara_et_al_2012": 5,
    "RBC_capitalstock_shock": 6,
    "SGU_2003": 7,
    "Gali_2008_chapter_2": 8,
    "Collard_2001_example1": 9,
    "McCandless_2008_Chapter_13": 10,
    "FV_et_al_2007_ABCD": 11,
    "RBC_baseline": 12,
    "RBC_state_dependent_GIRF": 13,
    "SGU_2004": 14,
    "Faia_2008": 15,
    "McCandless_2008_Chapter_9": 16,
}

def decode_env_name(env_name: str) -> int:
    prefix = env_name.rsplit('_', 1)[0]
    if prefix.endswith('_config'):
        prefix = prefix.removesuffix('_config')
    return 0
    return ENV_MAPPING[prefix]

def state_encoder(x):
    return x
def action_encoder(x):
    return x

class EconomicsDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing economic episodes data.

    This dataset handles variable-length economic episodes by loading state-action pairs
    from parquet files and preparing them for model training. The dataset performs:
    1. State-level padding/truncation to max_state_dim for uniform feature dimensions
    2. Sequence-level padding/truncation to max_seq_len for batch processing
    3. Generation of attention masks to handle variable-length sequences
    4. Task ID encoding for multi-task learning scenarios
    """

    def __init__(
        self, data_path: Path, max_state_dim: int, max_action_dim: int,
        max_endogenous_dim: int, max_model_params_dim: int, max_seq_len: int
    ):
        """
        Initialize the dataset with the given parameters.

        Args:
            data_path (Path): Path to the directory containing episode data files and metadata
            max_state_dim (int): Maximum dimension for state vectors after padding/truncation
            max_seq_len (int): Maximum sequence length for episodes (default: 512)
        """
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_endogenous_dim = max_endogenous_dim
        self.max_seq_len = max_seq_len
        self.max_model_params_dim = max_model_params_dim

        metadata_path = data_path / "metadata.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # todo: encode with llm
        # todo: get from dataset task id
        self.task_ids = [decode_env_name(item['env_name']) for item in self.metadata]

        # todo: Encoders of environment state and action
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder

        # todo: special words embeddings
        self.sw_state_embed = [...]
        self.sw_action_embed = [...]
        self.sw_reward_embed = [...]

    def __len__(self) -> int:
        """
        Get the total number of episodes in the dataset.

        Returns:
            int: Number of episodes in the dataset.
        """
        return len(self.metadata)

    @staticmethod
    def pad_sequence(sequence: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Pad or truncate a sequence to the specified length.

        Padding is added at the beginning of the sequence, which is useful for
        maintaining recent context in time-series data.

        Args:
            sequence (torch.Tensor): Input sequence of shape (seq_len, feature_dim)
            max_len (int): Target length for the sequence

        Returns:
            torch.Tensor: Padded or truncated sequence of shape (max_len, feature_dim)
        """
        seq_len = sequence.shape[0]
        if seq_len >= max_len:
            # If sequence is longer, take the last max_len elements
            return sequence[-max_len:]
        else:
            # If sequence is shorter, pad at the beginning
            padding = torch.zeros(
                max_len - seq_len,
                *sequence.shape[1:],
                dtype=sequence.dtype,
                )
            return torch.cat([padding, sequence], dim=0)  # Padding first, then sequence

    @staticmethod
    def pad_dim(sequence: torch.Tensor, max_dim: int) -> torch.Tensor:
        """
        Pad or truncate a sequence along the feature dimension.

        Used to ensure all state vectors have the same dimensionality.

        Args:
            sequence (torch.Tensor): Input sequence of shape (seq_len, feature_dim)
            max_dim (int): Maximum feature dimension to pad/truncate to

        Returns:
            torch.Tensor: Padded or truncated sequence of shape (seq_len, max_dim)
        """
        current_dim = sequence.shape[1]
        if current_dim >= max_dim:
            return sequence[:, :max_dim]  # truncate
        else:
            padding_size = max_dim - current_dim
            padding = torch.zeros(*sequence.shape[:-1], padding_size, dtype=sequence.dtype)
            return torch.cat([sequence, padding], dim=-1)

    def __getitem__(self, idx: int):
        """
        Get a single processed episode from the dataset.

        Processing steps:
        1. Load episode data from parquet file
        2. Convert states and actions to tensors
        3. Pad states to uniform feature dimension (max_state_dim)
        4. Pad sequences to uniform length (max_seq_len)
        5. Generate attention mask for valid positions

        Args:
            idx (int): Index of the episode to retrieve
x
        Returns:
            dict: A dictionary containing:
                - states (torch.Tensor): Padded state sequences [max_seq_len, max_state_dim]
                - actions (torch.Tensor): Padded action sequences [max_seq_len, action_dim]
                - task_id (torch.Tensor): Task identifier [scalar]
                - attention_mask (torch.Tensor): Boolean mask for valid positions [max_seq_len]
        """
        data = pd.read_parquet(self.metadata[idx]["output_dir"])

        states = torch.tensor(data['state'].tolist(), dtype=torch.float32)
        endogenous = torch.tensor(data['endogenous'].tolist(), dtype=torch.float32)
        actions = torch.tensor(data['action'].tolist(), dtype=torch.float32)
        rewards = torch.tensor(data['reward'].values, dtype=torch.float32).reshape(-1, 1)
        task_id = torch.tensor(self.task_ids[idx], dtype=torch.long)

        info = data.iloc[0]["info"]
        model_params = info["model_params"]

        sorted_model_params = list(sorted(model_params.items()))
        model_params_values = torch.tensor([v for k, v in sorted_model_params] + [0] * (self.max_model_params_dim - len(sorted_model_params)), dtype=torch.float32)

        # Pad states to max_state_dim
        states = self.pad_dim(states, self.max_state_dim)
        state_description = data.iloc[0]["info"]["state_description"]
        action_description = data.iloc[0]["info"]["action_description"]
        endogenous_description = data.iloc[0]["info"]["endogenous_description"]
        states_info = torch.tensor([STATE_MAPPING[state] for state in state_description] + [0] * (self.max_state_dim - len(state_description)), dtype=torch.long)
        actions_info = torch.tensor([ACTION_MAPPING[action] for action in action_description] + [0] * (self.max_action_dim - len(action_description)), dtype=torch.long)
        endogenous_info = torch.tensor([STATE_MAPPING[endogenous] for endogenous in endogenous_description] + [0] * (self.max_endogenous_dim - len(endogenous_description)), dtype=torch.long)
        assert len(states_info) == self.max_state_dim, f"states_info length is {len(states_info)} but max_state_dim is {self.max_state_dim}"
        assert len(actions_info) == self.max_action_dim, f"actions_info length is {len(actions_info)} but max_action_dim is {self.max_action_dim}"
        # Pad actions to max_actions_dim
        actions = self.pad_dim(actions, self.max_action_dim)
        endogenous = self.pad_dim(endogenous, self.max_endogenous_dim)

        # Get original sequence length
        orig_seq_len = len(states)

        # Pad sequences to max_seq_len
        states = self.pad_sequence(states, self.max_seq_len)
        actions = self.pad_sequence(actions, self.max_seq_len)
        rewards = self.pad_sequence(rewards, self.max_seq_len)
        endogenous = self.pad_sequence(endogenous, self.max_seq_len)

        # Create attention mask
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        attention_mask[:min(orig_seq_len, self.max_seq_len)] = True

        return {
            'states': states,  # [max_seq_len, max_state_dim]
            'states_info': states_info,  # [max_state_dim]
            'actions': actions,  # [max_seq_len, action_dim]
            'actions_info': actions_info,  # [action_dim]
            'endogenous': endogenous,  # [max_seq_len, max_endogenous_dim]
            'endogenous_info': endogenous_info,  # [max_endogenous_dim]
            'reward': rewards,  # [max_seq_len, 1]
            'task_id': task_id,  # scalar
            'model_params': model_params_values,
            'attention_mask': attention_mask  # [max_seq_len]
        }
