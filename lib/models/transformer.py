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
            num_layers: int = 4
    ):
        super().__init__()
        self.state_embedding = nn.Linear(state_dim, d_model)
        self.task_embedding = nn.Embedding(num_tasks, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, states: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            states (torch.Tensor): Batch of state sequences [batch_size, seq_length, state_dim]
            task_id (torch.Tensor): Batch of task identifiers [batch_size]

        Returns:
            torch.Tensor: Predicted actions for each state [batch_size, seq_length, action_dim]
        """
        state_emb = self.state_embedding(states)  # [bs, seq_len, d_model]
        task_emb = self.task_embedding(task_id)  # [bs, d_model]

        inp = torch.cat([task_emb, state_emb], dim=1)  # [bs, seq_len+1, d_model]
        inp = inp.transpose(0, 1)
        # inp = self.positional_encoding(inp)

        encoded = self.transformer(inp)  # [seq_len+1, bs, d_model]
        encoded = encoded[1:]  # [seq_len, bs, d_model]
        encoded = encoded.transpose(0, 1)  # [bs, seq_len, d_model]

        actions_pred = self.action_head(encoded)  # [bs, seq_len, action_dim]

        return actions_pred
