import math
from lib.dataset import ACTION_MAPPING, STATE_MAPPING
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
        self.d_model = d_model * (1 + state_dim + action_dim + 1)

        self.state_embedding = nn.Embedding(len(STATE_MAPPING), d_model - 1)
        self.action_embedding = nn.Embedding(len(ACTION_MAPPING), d_model - 1)
        self.reward_embedding = nn.Linear(1, d_model, dtype=torch.float32)  # Assuming scalar rewards
        self.task_embedding = nn.Embedding(num_tasks, d_model)

        # Add token type embedding
        # 4 types: task(0), state(1), action(2), reward(3)
        # self.token_type_embedding = nn.Embedding(4, d_model)

        # self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
               d_model=self.d_model,
                nhead=nhead,
                dtype=torch.float32
            ), 
            num_layers=num_layers
        )
        self.action_head = nn.Linear(self.d_model, action_dim, dtype=torch.float32)

    def get_state_embedding(self, states: torch.Tensor, states_info: torch.Tensor) -> torch.Tensor:
        # states: [batch_size, seq_length, state_dim]
        # states_info: [batch_size, state_dim]
        
        # Get embeddings for state classes [batch_size, state_dim, d_model-1]
        class_embeddings = self.state_embedding(states_info)
        
        # Reshape states to [batch_size, seq_len, state_dim] and expand class embeddings
        # Expand class embeddings to match sequence length
        class_embeddings = class_embeddings.unsqueeze(1).expand(-1, states.shape[1], -1, -1)
        states = states.unsqueeze(-1)
        
        # Concatenate states with class embeddings along last dimension
        # [batch_size, seq_len, state_dim, d_model]
        combined = torch.cat([class_embeddings, states], dim=-1)
        
        # Flatten the state_dim dimension into d_model
        # [batch_size, seq_len, d_model] 
        return combined.view(combined.shape[0], combined.shape[1], -1)

    def get_action_embedding(self, actions: torch.Tensor, actions_info: torch.Tensor) -> torch.Tensor:
        # actions: [batch_size, seq_length, action_dim]
        # actions_info: [batch_size, action_dim]
        class_embeddings = self.action_embedding(actions_info)
        class_embeddings = class_embeddings.unsqueeze(1).expand(-1, actions.shape[1], -1, -1)
        actions = actions.unsqueeze(-1)
        combined = torch.cat([class_embeddings, actions], dim=-1)  
        return combined.view(combined.shape[0], combined.shape[1], -1)
        
    def forward(
            self,
            states: torch.Tensor,
            states_info: torch.Tensor,
            actions: torch.Tensor = None,
            actions_info: torch.Tensor = None,
            rewards: torch.Tensor = None,
            task_ids: torch.Tensor = None,
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
        seq_length = states.shape[1]

        # Embed state and task
        state_emb = self.get_state_embedding(states, states_info)
        task_emb = self.task_embedding(task_ids).unsqueeze(1)  # [bs, 1, d_model]    
        # Sequence will only include task + states and actions until the current state
        # if actions is None or rewards is None:
        #     sequence = torch.cat([task_emb, state_emb[:, 0:1]], dim=1)  # [bs, 2, d_model]
        #     token_types = torch.tensor([0, 1], dtype=torch.long, device=states.device).unsqueeze(0).repeat(batch_size, 1)
        # else:
        action_emb = self.get_action_embedding(actions, actions_info)
        reward_emb = self.reward_embedding(rewards)

        # Create sequence: [task, state_1, action_1, reward_1, state_2, ...]
        sequence = torch.cat([
            task_emb.repeat(1, seq_length, 1), # [bs, seq_length, d_model]
            state_emb, # [bs, seq_length, d_model * state_dim]
            action_emb, # [bs, seq_length, d_model * action_dim]
            reward_emb # [bs, seq_length, d_model]
        ], dim=2)

        # token_type_emb = self.token_type_embedding(token_types)
        sequence = sequence # + token_type_emb

        # Apply transformer
        encoded = self.transformer(sequence.transpose(0, 1)).transpose(0, 1)  # [batch_size, seq_len, d_model]

        # Extract only state positions for action prediction
        # if actions is None or rewards is None:
        #     state_positions = encoded[:, 1:]  # Skip task token, only predict for the state
        # else:
            # Extract positions after each state token (every 3rd position after task token)
    
        # Predict actions for all states
        actions_pred = self.action_head(encoded)[:, :-1, :] # [bs, seq_length-1, action_dim]
        return actions_pred
