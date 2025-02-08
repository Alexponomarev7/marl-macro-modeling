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

    This model processes sequences of states and task identifiers to predict actions.
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
            actions: torch.Tensor,
            rewards: torch.Tensor,
            task_id: torch.Tensor,
            padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass creating interleaved sequences of states, actions, and rewards.

        Args:
            states: [batch_size, seq_length, state_dim]
            actions: [batch_size, seq_length, action_dim]
            rewards: [batch_size, seq_length, 1]
            task_id: [batch_size]
            padding_mask: [batch_size, seq_length]

        Returns:
            torch.Tensor: Predicted next actions [batch_size, seq_length, action_dim]
        """
        batch_size, seq_length = states.shape[:2]

        state_emb = self.state_embedding(states)  # [bs, seq_len, d_model]
        action_emb = self.action_embedding(actions)  # [bs, seq_len, d_model]
        reward_emb = self.reward_embedding(rewards)   # [bs, seq_len, d_model]
        task_emb = self.task_embedding(task_id).unsqueeze(1)  # [bs, 1, d_model]

        print("state_emb.shapeb", state_emb.shape)
        print("action_emb.shape", action_emb.shape)
        print("reward_emb.shape", reward_emb.shape)
        print("task_emb.shape", task_emb.shape)

        # Create token type IDs for each position in the sequence
        # Repeat pattern: [task(0), state(1), action(2), reward(3), state(1), action(2), reward(3), ...]
        token_types = torch.zeros(batch_size, 1 + seq_length * 3, dtype=torch.long, device=states.device)
        token_types[:, 1::3] = 1  # states
        token_types[:, 2::3] = 2  # actions
        token_types[:, 3::3] = 3  # rewards
        token_type_emb = self.token_type_embedding(token_types)

        # Interleave the sequence: [task_emb, state_1, action_1, reward_1, state_2, ...]
        # Initialize sequence
        sequence = torch.zeros(
            batch_size,
            1 + seq_length * 3,
            state_emb.size(-1),
            device=states.device
        )

        # Add task embedding at the start
        sequence[:, 0] = task_emb.squeeze(1)

        # Add the rest of the sequence
        for idx, i in enumerate(range(seq_length)):
            # print("idx", idx)
            # print("sequence.shape", sequence.shape)
            # print("action_emb.shape", action_emb.shape)
            base_idx = 1 + i * 3  # Calculate base index for each triplet
            sequence[:, base_idx] = state_emb[:, i, :]
            sequence[:, base_idx + 1] = action_emb[:, i, :]
            sequence[:, base_idx + 2] = reward_emb[:, i, :]

        # Add token type embeddings
        sequence = sequence + token_type_emb

        # Update padding mask if provided
        if padding_mask is not None:
            # Expand padding mask to account for interleaved sequence
            expanded_mask = torch.zeros(
                batch_size,
                1 + seq_length * 3,
                dtype=torch.bool,
                device=padding_mask.device
            )
            expanded_mask[:, 0] = False  # Always attend to task token
            for i in range(seq_length):
                idx = 1 + i * 3
                expanded_mask[:, idx:idx + 3] = padding_mask[:, i:i + 1].repeat(1, 3)
            padding_mask = expanded_mask

        # Process through transformer
        sequence = sequence.transpose(0, 1)  # [1 + seq_len*3, bs, d_model]
        encoded = self.transformer(sequence, src_key_padding_mask=padding_mask)  # [1 + seq_len*3, bs, d_model]

        # Extract state positions for action predictions
        encoded = encoded.transpose(0, 1)  # [bs, 1 + seq_len*3, d_model]
        state_positions = encoded[:, 1::3]  # Take embeddings after state tokens

        # Predict actions
        actions_pred = self.action_head(state_positions)  # [bs, seq_len, action_dim]

        return actions_pred
