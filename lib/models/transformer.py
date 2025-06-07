import math
from lib.dataset import ACTION_MAPPING, STATE_MAPPING
from lib.envs.environment_base import AbstractEconomicEnv
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
        d_model: int,
        nhead: int,
        num_layers: int,
        max_seq_len: int,
        model_params_dim: int,
        pinn_output_dim: int,  # Optional PINN head output dimension
        has_pinn: bool,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len
        self.d_model = d_model * (1 + state_dim + action_dim + 1)
        self.has_pinn = has_pinn
        self.model_params_dim = model_params_dim

        self.state_embedding = nn.Embedding(len(STATE_MAPPING), d_model - 1)
        self.action_embedding = nn.Embedding(len(ACTION_MAPPING), d_model - 1)
        self.reward_embedding = nn.Linear(1, d_model, dtype=torch.float32)  # Assuming scalar rewards
        self.task_embedding = nn.Embedding(num_tasks, d_model - model_params_dim)

        # Add token type embedding
        # 4 types: task(0), state(1), action(2), reward(3)
        # self.token_type_embedding = nn.Embedding(4, d_model)

        # self.positional_encoding = PositionalEncoding(d_model)
        
        # Create causal mask to ensure transformer only looks at past tokens
        self.register_buffer('causal_mask', torch.triu(torch.ones(2048, 2048), diagonal=1).bool())
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
               d_model=self.d_model,
                nhead=nhead,
                dtype=torch.float32
            ), 
            num_layers=num_layers
        )
        self.action_head = nn.Linear(self.d_model, action_dim, dtype=torch.float32)

        # Optional PINN head for predicting additional data
        if self.has_pinn:
            self.pinn_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, pinn_output_dim)
            )

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
        # print(class_embeddings.shape)
        # print(actions.shape)
        # assert False
        combined = torch.cat([class_embeddings, actions], dim=-1)  
        return combined.view(combined.shape[0], combined.shape[1], -1)
        
    def forward(
        self,
        states: torch.Tensor,
        states_info: torch.Tensor,
        actions: torch.Tensor,
        actions_info: torch.Tensor,
        rewards: torch.Tensor,
        task_ids: torch.Tensor,
        model_params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass creating sequences of states, actions, and rewards, and predicts actions for each timestep.

        Args:
            states: [batch_size, seq_length, state_dim]
            actions: [batch_size, seq_length-1, action_dim] or None (for first step)
            rewards: [batch_size, seq_length-1, 1] or None (for first step)
            task_id: [batch_size]
            padding_mask: [batch_size, seq_length]

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Predicted actions and optional PINN predictions
        """
        seq_length = states.shape[1]

        # Embed state and task
        state_emb = self.get_state_embedding(states, states_info)

        task_emb = torch.cat([
            self.task_embedding(task_ids).unsqueeze(1),  # [bs, 1, d_model] 
            model_params.unsqueeze(1)  # [bs, 1, num_params]
        ], dim=2)  # [bs, 1, d_model + num_params]
        
        action_emb = self.get_action_embedding(actions, actions_info)
        reward_emb = self.reward_embedding(rewards)

        # Create sequence: [task, state_1, action_1, reward_1, state_2, ...]
        sequence = torch.cat([
            task_emb.repeat(1, seq_length, 1), # [bs, seq_length, d_model]
            state_emb, # [bs, seq_length, d_model * state_dim]
            action_emb, # [bs, seq_length, d_model * action_dim]
            reward_emb # [bs, seq_length, d_model]
        ], dim=2)
        sequence = sequence # + token_type_emb

        mask = self.causal_mask[:seq_length, :seq_length]
        encoded = self.transformer(sequence.transpose(0, 1), mask=mask).transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Main action prediction head
        actions_pred = self.action_head(encoded) # [bs, seq_length-1, action_dim]
        
        # Optional PINN predictions
        pinn_pred = None
        if self.has_pinn:
            pinn_pred = self.pinn_head(encoded) # [bs, seq_length-1, pinn_output_dim]

        return actions_pred, pinn_pred

    def _get_state_info(self, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        state_values, state_ids = [], []
        for state_name, state_value in state.items():
            assert state_name in STATE_MAPPING, f"State {state_name} not found in STATE_MAPPING"
            state_ids.append(STATE_MAPPING[state_name])
            state_values.append(state_value)
        
        state_values += [0] * (self.state_dim - len(state_values))
        state_ids += [STATE_MAPPING["Empty"]] * (self.state_dim - len(state_ids))
        return torch.tensor(state_values, dtype=torch.float32), torch.tensor(state_ids, dtype=torch.long)

    def _get_action_info(self, action: dict) -> tuple[torch.Tensor, torch.Tensor]:
        action_values, action_ids = [], []
        for action_name, action_value in action.items():
            assert action_name in ACTION_MAPPING, f"Action {action_name} not found in ACTION_MAPPING"
            action_ids.append(ACTION_MAPPING[action_name])
            action_values.append(action_value)
        action_values += [0] * (self.action_dim - len(action_values))
        action_ids += [ACTION_MAPPING["Empty"]] * (self.action_dim - len(action_ids))
        return torch.tensor(action_values, dtype=torch.float32), torch.tensor(action_ids, dtype=torch.long)

    def inference(self, env: AbstractEconomicEnv, max_steps: int = 50) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        init_state, _ = env.reset()
        init_state_values, states_info = self._get_state_info({
            state_name: init_state[state_name] for state_name in env.state_description.keys()
        })
        init_action_values, actions_info = self._get_action_info(
            {k: 0.0 for k, _ in env.action_description.items()}
        )

        state_to_plot = [{
            state_name: init_state[state_name] for state_name in env.state_description.keys()
        }]
        action_to_plot = [{k: 0.0 for k, _ in env.action_description.items()}]

        state_history = [init_state_values]
        action_history = [init_action_values]
        reward_history = [torch.tensor([0.0], dtype=torch.float32)]
        task_ids = torch.tensor([env.task_id], dtype=torch.long)
        model_params = torch.tensor([v for _, v in sorted(env.params.items())] + [0] * (self.model_params_dim - len(env.params)), dtype=torch.float32)

        for _ in range(max_steps):
            out, _ = self.forward(
                states=torch.stack(state_history).unsqueeze(0).to(self.device),
                states_info=states_info.unsqueeze(0).to(self.device),
                actions=torch.stack(action_history).unsqueeze(0).to(self.device),
                actions_info=actions_info.unsqueeze(0).to(self.device),
                rewards=torch.stack(reward_history).unsqueeze(0).to(self.device),
                task_ids=task_ids.to(self.device),
                model_params=model_params.unsqueeze(0).to(self.device)
            )

            action = float(out[0][-1][0])
            next_state, reward, _, _, _ = env.step(action) # type: ignore
            state_to_plot.append({
                state_name: next_state[state_name] for state_name in env.state_description.keys()
            })
            action_to_plot.append({k: next_state[k] for k, _ in env.action_description.items()})

            state_history.append(self._get_state_info(state_to_plot[-1])[0])
            action_history.append(self._get_action_info(action_to_plot[-1])[0])
            reward_history.append(torch.tensor([reward], dtype=torch.float32))

        return state_to_plot, action_to_plot
