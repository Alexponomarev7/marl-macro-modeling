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

            # Create sequence: [task, state_1, action_1, reward_1, state_2, ...]
            sequence_list = [task_emb]
            token_types_list = [torch.zeros(batch_size, 1, dtype=torch.long, device=states.device)]  # task token

            for i in range(seq_length - 1):
                # todo: optimize
                sequence_list.extend([
                    state_emb[:, i:i+1],
                    action_emb[:, i:i+1],
                    reward_emb[:, i:i+1]
                ])
                token_types_list.extend([
                    torch.ones(batch_size, 1, dtype=torch.long, device=states.device),      # state
                    2 * torch.ones(batch_size, 1, dtype=torch.long, device=states.device),  # action
                    3 * torch.ones(batch_size, 1, dtype=torch.long, device=states.device)   # reward
                ])

            # Add final state
            sequence_list.append(state_emb[:, -1:])
            token_types_list.append(torch.ones(batch_size, 1, dtype=torch.long, device=states.device))

            sequence = torch.cat(sequence_list, dim=1)
            token_types = torch.cat(token_types_list, dim=1)

        # Add token type embeddings
        token_type_emb = self.token_type_embedding(token_types)
        sequence = sequence + token_type_emb

        # Apply transformer
        encoded = self.transformer(sequence.transpose(0, 1)).transpose(0, 1)  # [batch_size, seq_len, d_model]

        # Extract only state positions for action prediction
        if actions is None or rewards is None:
            state_positions = encoded[:, 1:]  # Skip task token, only predict for the state
        else:
            # Extract positions after each state token (every 3rd position after task token)
            state_indices = torch.arange(1, encoded.size(1), 3, device=encoded.device)
            state_positions = encoded[:, state_indices]

        # Predict actions for all states
        actions_pred = self.action_head(state_positions)  # [batch_size, num_states, action_dim]

        return actions_pred
