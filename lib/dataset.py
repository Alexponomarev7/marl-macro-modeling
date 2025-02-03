import json
import torch
import pandas as pd
from pathlib import Path


class Dataset:
    """
    A PyTorch Dataset for processing economic episodes into fixed-length vectors.

    Each episode consists of a sequence of (state, action, reward) steps. The dataset:
    1. Loads raw episode data from parquet files
    2. Processes each step's components into fixed-size vectors
    3. Flattens the entire episode into a single vector
    4. Ensures all episodes are exactly max_sequence_length by padding/truncating

    This creates a uniform interface for training models on variable-length episodes
    by converting them all to fixed-length vectors.
    """

    def __init__(self, data_path: Path, max_state_dim: int, max_sequence_length: int = 512):
        """
        Initialize dataset parameters and load metadata.

        Args:
            data_path (Path): Directory containing episode data files and metadata
            max_state_dim (int): Maximum dimension for state vectors after padding/truncating
            max_sequence_length (int): Fixed length for output episode vectors
        """
        self.max_state_dim = max_state_dim
        self.max_sequence_length = max_sequence_length

        metadata_path = data_path / "metadata.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # todo: Encoders of environment state and action
        self.state_encoder = lambda x: x
        self.action_encoder = lambda x: x

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
    def pad_sequence(sequence: torch.Tensor, max_dim: int, dim: int = 1) -> torch.Tensor:
        """
        Pad or truncate a tensor to a fixed size along specified dimension.

        Args:
            sequence (torch.Tensor): Input tensor to be padded/truncated
            max_dim (int): Target size for the specified dimension
            dim (int): Which dimension to pad/truncate (0=sequence, 1=features)

        Returns:
            torch.Tensor: Tensor with specified dimension exactly equal to max_dim
        """
        if dim == 1:
            current_dim = sequence.shape[1]
            if current_dim >= max_dim:
                return sequence[:, :max_dim]  # truncate
            else:
                padding_size = max_dim - current_dim
                padding = torch.zeros(*sequence.shape[:-1], padding_size, dtype=sequence.dtype)
                return torch.cat([sequence, padding], dim=-1)
        else:  # dim == 0
            current_len = sequence.shape[0]
            if current_len >= max_dim:
                return sequence[:max_dim]  # truncate
            else:
                padding_size = max_dim - current_len
                padding = torch.zeros(padding_size, sequence.shape[1], dtype=sequence.dtype)
                return torch.cat([sequence, padding], dim=0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert an episode into a fixed-length vector representation.

        Processing pipeline:
        1. Load episode data from parquet file
        2. Convert states, actions, rewards to tensors
        3. Pad/truncate state vectors to max_state_dim
        4. Concatenate (state, action, reward) for each timestep
        5. Flatten entire episode into single vector
        6. Pad/truncate to exactly max_sequence_length

        Args:
            idx (int): Index of episode to process

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - padded_episode: Fixed-length vector of shape (max_sequence_length,)
                  containing the flattened episode data
                - padded_mask: Boolean tensor of shape (max_sequence_length,) marking
                  valid data (True) versus padding (False)
        """
        data = pd.read_parquet(self.metadata[idx]["output_dir"])

        # TODO(aponomarev): potential bug with order
        states = torch.tensor(data['state'].tolist(), dtype=torch.float32)
        actions = torch.tensor(data['action'].tolist(), dtype=torch.float32)
        rewards = torch.tensor(data['reward'].values, dtype=torch.float32).reshape(-1, 1)
        # First pad the states to max_state_dim
        padded_states = self.pad_sequence(states, self.max_state_dim, dim=1)

        return padded_states, actions, torch.tensor([0])
        # Concatenate states, actions, and rewards for each step
        # print("HERE")
        # print(padded_states.shape, actions.shape, rewards.shape)
        # step_embeds = torch.cat([
        #     padded_states,
        #     actions,
        #     rewards
        # ], dim=1)

        # # Flatten the entire episode into a single vector
        # flattened_episode = step_embeds.reshape(-1)
        # episode_mask = torch.ones_like(flattened_episode, dtype=torch.bool)

        # # Create output tensors of exact size max_sequence_length
        # padded_episode = torch.zeros(self.max_sequence_length, dtype=torch.float32)
        # padded_mask = torch.zeros(self.max_sequence_length, dtype=torch.bool)

        # # Copy actual data (truncating or padding as needed)
        # length = min(len(flattened_episode), self.max_sequence_length)
        # padded_episode[:length] = flattened_episode[:length]
        # padded_mask[:length] = episode_mask[:length]

        # return padded_episode, padded_mask
