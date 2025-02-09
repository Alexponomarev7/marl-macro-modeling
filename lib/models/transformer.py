import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for transformer inputs.

    This module adds positional information to the input embeddings using
    sine and cosine functions of different frequencies.

    Args:
        d_model (int): The dimension of the model's embeddings
        max_len (int, optional): Maximum sequence length. Defaults to 5000.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [seq_len, batch_size, d_model]

        Returns:
            torch.Tensor: Input with positional encoding added
        """
        return x + self.pe[:x.size(0)]


class AlgorithmDistillationTransformer(nn.Module):
    """
    A transformer-based model for algorithm distillation.

    This model processes sequences of states and task identifiers to predict the next action.
    It uses a transformer architecture with positional encoding.

    Args:
        state_dim (int): Dimension of the state space
        action_dim (int): Dimension of the action space
        num_tasks (int): Number of different tasks
        d_model (int, optional): Dimension of the model's embeddings. Defaults to 128
        nhead (int, optional): Number of attention heads. Defaults to 4
        num_layers (int, optional): Number of transformer layers. Defaults to 4
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            num_tasks: int,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 4,
            max_seq_len: int = 512,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len

        self.state_embedding = nn.Linear(state_dim, d_model, dtype=torch.float32)
        self.action_embedding = nn.Linear(action_dim, d_model, dtype=torch.float32)
        self.reward_embedding = nn.Linear(1, d_model, dtype=torch.float32)  # Assuming scalar rewards
        self.task_embedding = nn.Embedding(num_tasks, d_model)

        # Add token type embedding
        # 4 types: task(0), state(1), action(2), reward(3)
        self.token_type_embedding = nn.Embedding(4, d_model)

        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dtype=torch.float32)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim, dtype=torch.float32)

    def forward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor = None,
            rewards: torch.Tensor = None,
            task_id: torch.Tensor = None,
            padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass creating sequences of states, actions, and rewards, and predicts actions for each timestep.

        Args:
            states: [batch_size, seq_length, state_dim]
            actions: [batch_size, seq_length-1, action_dim] or None (for first step)
            rewards: [batch_size, seq_length-1, 1] or None (for first step)
            task_id: [batch_size]
            padding_mask: [batch_size, seq_length]

        Returns:
            torch.Tensor: Predicted actions [batch_size, seq_length, action_dim]
        """
        batch_size, seq_length = states.shape[:2]

        # Embed state and task
        state_emb = self.state_embedding(states)
        task_emb = self.task_embedding(task_id).unsqueeze(1)  # [bs, 1, d_model]

        # Sequence will only include task + states and actions until the current state
        if actions is None or rewards is None:
            sequence = torch.cat([task_emb, state_emb[:, 0:1]], dim=1)  # [bs, 2, d_model]
            token_types = torch.tensor([0, 1], dtype=torch.long, device=states.device).unsqueeze(0).repeat(batch_size, 1)
        else:
            action_emb = self.action_embedding(actions)
            reward_emb = self.reward_embedding(rewards)

            # Interleave task, states, actions, and rewards
            sequence = torch.zeros(batch_size, 1 + seq_length * 3, state_emb.size(-1), device=states.device)
            sequence[:, 0] = task_emb.squeeze(1)
            for i in range(seq_length - 1):
                base_idx = 1 + i * 3
                sequence[:, base_idx] = state_emb[:, i]
                sequence[:, base_idx + 1] = action_emb[:, i]
                sequence[:, base_idx + 2] = reward_emb[:, i]

            # Add final state
            sequence[:, 1 + (seq_length - 1) * 3] = state_emb[:, -1]

            # Create token type IDs
            token_types = torch.zeros(batch_size, 1 + seq_length * 3, dtype=torch.long, device=states.device)
            token_types[:, 1::3] = 1  # states
            token_types[:, 2::3] = 2  # actions
            token_types[:, 3::3] = 3  # rewards

        # Add token type embeddings
        token_type_emb = self.token_type_embedding(token_types)
        sequence = sequence + token_type_emb

        # Positional encoding
        # sequence = self.positional_encoding(sequence.transpose(0, 1))

        # Transformer
        encoded = self.transformer(sequence).transpose(0, 1)  # [batch_size, seq_len, d_model]

        # Extract state positions for action prediction
        state_positions = []
        if actions is None or rewards is None:
            state_positions = encoded[:, -1:]  # Only predict for the last position
        else:
            # Extract all state positions (every third position after task token)
            state_indices = torch.arange(1, encoded.size(1), 3, device=encoded.device)
            state_positions = encoded[:, state_indices]

        # Predict actions for all states
        actions_pred = self.action_head(state_positions)  # [batch_size, seq_length, action_dim]

        print("actions_pred.shape", actions_pred.shape)
        print("batch_size", batch_size)
        print("seq_length", seq_length)

        return actions_pred