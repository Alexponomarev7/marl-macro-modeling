import json
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class EconomicsDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing economic episodes data.

    This dataset handles variable-length economic episodes by padding or truncating
    them to a fixed length. Each episode consists of states, actions, and rewards
    that are combined into a single embedding tensor. The padding is applied at two levels:
    1. State-level padding: Each state vector is padded/truncated to max_state_dim
    2. Sequence-level padding: The entire sequence is padded/truncated to max_total_dim
    """

    def __init__(self, data_path: Path, max_state_dim: int, max_total_dim: int = 512):
        """
        Initialize the dataset with the given path and maximum lengths.

        Args:
            data_path (Path): Path to the directory containing the dataset files.
            max_state_dim (int, optional): Maximum dimension for state vectors. Defaults to 50.
            max_total_dim (int, optional): Maximum total dimension (state+action+reward). Defaults to 512.
        """
        self.max_state_dim = max_state_dim
        self.max_total_dim = max_total_dim

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
    def pad_sequence(sequence: torch.Tensor, max_dim: int) -> torch.Tensor:
        """
        Pad or truncate a sequence to the specified maximum dimension.

        Args:
            sequence (torch.Tensor): Input sequence tensor of shape (seq_len, feature_dim).
            max_dim (int): Maximum dimension to pad/truncate to.

        Returns:
            torch.Tensor: Padded or truncated sequence
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
        Get a single episode from the dataset.

        The episode is processed by:
        1. Loading the parquet file
        2. Converting states, actions, and rewards to tensors
        3. Padding states to max_state_len (along feature dimension)
        4. Concatenating them into a single embedding
        5. Padding the entire sequence to max_sequence_len (along time dimension)
        6. Creating an attention mask for the sequence padding

        Args:
            idx (int): Index of the episode to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - padded_embeds: Tensor of shape (max_sequence_len, feature_dim) containing
                  the padded sequence of state-action-reward embeddings
                - mask: Boolean tensor of shape (max_sequence_len,) indicating valid (True)
                  and padded (False) positions
        """
        data = pd.read_parquet(self.metadata[idx]["output_dir"])

        # state_embed = self.state_encoder(data['state_desc']) # llm encoded
        # action_embed = self.action_encoder(data['action_desc']) # llm encoded
        # environment_embed = torch.concat([state_embed, action_embed], dim=1)

        # todo: change states order
        states = torch.tensor(data['state'].tolist(), dtype=torch.float32)
        actions = torch.tensor(data['action'].tolist(), dtype=torch.float32)
        rewards = torch.tensor(data['reward'].values, dtype=torch.float32).reshape(-1, 1)

        # Get dimensions
        original_state_dim = states.shape[1]
        action_dim = actions.shape[1]

        # First pad the states to max_state_len (feature dimension padding)
        padded_states = self.pad_sequence(states, self.max_state_dim)

        # TODO(aponomarev): fix
        return padded_states, actions, torch.tensor([0])
        # embeds = torch.cat([
        #     # environment_embed,
        #     # self.sw_state_embed,
        #     padded_states,
        #     # self.sw_action_embed,
        #     actions,
        #     # self.sw_reward_embed,
        #     rewards

        # ], dim=1)

        # padded_embeds = self.pad_sequence(embeds, self.max_total_dim)

        # # Create mask for valid dimensions
        # mask = torch.zeros(padded_embeds.shape[0], padded_embeds.shape[1], dtype=torch.bool)

        # # Mark valid state dimensions
        # mask[:, :original_state_dim] = True
        # # Mark valid action dimensions
        # mask[:, self.max_state_dim:self.max_state_dim + action_dim] = True
        # # Mark valid reward dimension
        # mask[:, self.max_state_dim + action_dim] = True

        # return padded_embeds, mask