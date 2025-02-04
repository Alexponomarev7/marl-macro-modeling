import json
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


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

    def __init__(self, data_path: Path, max_state_dim: int, max_seq_len: int = 512):
        """
        Initialize the dataset with the given parameters.

        Args:
            data_path (Path): Path to the directory containing episode data files and metadata
            max_state_dim (int): Maximum dimension for state vectors after padding/truncation
            max_seq_len (int): Maximum sequence length for episodes (default: 512)
        """
        self.max_state_dim = max_state_dim
        self.max_seq_len = max_seq_len

        metadata_path = data_path / "metadata.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # todo: encode with llm
        self.task_ids = [item.get('task_id', 0) for item in self.metadata]

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

        Returns:
            dict: A dictionary containing:
                - states (torch.Tensor): Padded state sequences [max_seq_len, max_state_dim]
                - actions (torch.Tensor): Padded action sequences [max_seq_len, action_dim]
                - task_id (torch.Tensor): Task identifier [scalar]
                - attention_mask (torch.Tensor): Boolean mask for valid positions [max_seq_len]
        """
        data = pd.read_parquet(self.metadata[idx]["output_dir"])

        states = torch.tensor(data['state'].tolist(), dtype=torch.float32)
        actions = torch.tensor(data['action'].tolist(), dtype=torch.float32)
        # rewards = torch.tensor(data['reward'].values, dtype=torch.float32).reshape(-1, 1)
        task_id = torch.tensor(self.task_ids[idx], dtype=torch.long)

        # Pad states to max_state_dim
        states = self.pad_dim(states, self.max_state_dim)

        # Get original sequence length
        orig_seq_len = len(states)

        # Pad sequences to max_seq_len
        states = self.pad_sequence(states, self.max_seq_len)
        actions = self.pad_sequence(actions, self.max_seq_len)

        # Create attention mask
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        attention_mask[:min(orig_seq_len, self.max_seq_len)] = True

        return {
            'states': states,  # [max_seq_len, max_state_dim]
            'actions': actions,  # [max_seq_len, action_dim]
            'task_id': task_id,  # scalar
            'attention_mask': attention_mask  # [max_seq_len]
        }