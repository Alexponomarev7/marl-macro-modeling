import inspect
import math

import gymnasium as gym
import numpy as np
from lib.dataset import Tokenizer
from lib.envs.environment_base import AbstractEconomicEnv
import torch
import torch.nn as nn


def _to_scalar(x) -> float:
    """Env state values are floats or shape-(1,) arrays."""
    return float(np.asarray(x, dtype=np.float64).reshape(-1)[0])


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


DELTA_SCALE = 100.0  # first differences are scaled into symlog's near-linear range


def _first_difference(x: torch.Tensor) -> torch.Tensor:
    """x_t - x_{t-1} along the sequence dim ([batch, seq, ...]), 0 at the first step."""
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], dim=1)


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
        context_only: bool = False,
    ):
        """context_only: hide the task id and the model parameters."""
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len
        self.d_model = d_model * (1 + state_dim + action_dim + 1)
        self.has_pinn = has_pinn
        self.model_params_dim = model_params_dim
        self.context_only = context_only

        self.tokenizer = Tokenizer()
        # slot = [variable-name embedding | numeric channels]
        self.state_embedding = nn.Embedding(self.tokenizer.num_state_tokens, d_model - 2)
        self.action_embedding = nn.Embedding(self.tokenizer.num_action_tokens, d_model - 2)
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
                norm_first=True,
                dtype=torch.float32
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(self.d_model),
            enable_nested_tensor=False,
        )
        self.action_head = nn.Linear(self.d_model, action_dim, dtype=torch.float32)
        # zero-init residual head: training starts from persistence, a_t = a_{t-1}
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

        # Optional PINN head for predicting additional data
        if self.has_pinn:
            self.pinn_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, pinn_output_dim)
            )

    @staticmethod
    def _slot_embedding(embedding: nn.Embedding, info: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
        """info: [batch, slots]; channels: [batch, seq, slots, C] -> [batch, seq, slots * d_model]."""
        names = embedding(info).unsqueeze(1).expand(-1, channels.shape[1], -1, -1)
        combined = torch.cat([names, channels], dim=-1)
        return combined.reshape(combined.shape[0], combined.shape[1], -1)

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
            states: [batch_size, seq_length, state_dim] - s_t
            actions: [batch_size, seq_length, action_dim] - a_{t-1} (0 at an episode's first step)
            rewards: [batch_size, seq_length, 1] - r_{t-1} (0 at an episode's first step)
            task_ids: [batch_size]

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Predicted actions and optional PINN predictions
        """
        seq_length = states.shape[1]
        states, actions, rewards, model_params = (
            torch.nan_to_num(x, nan=0.0, posinf=1000.0, neginf=-1000.0).clamp(-1000.0, 1000.0)
            for x in (states, actions, rewards, model_params)
        )
        prev_actions = actions  # step t carries a_{t-1}
        state_channels = torch.stack([symlog(states), symlog(DELTA_SCALE * _first_difference(states))], -1)
        action_channels = torch.stack([symlog(actions), symlog(DELTA_SCALE * _first_difference(actions))], -1)
        reward_input = symlog(rewards)
        model_params = symlog(model_params)

        # Embed state and task
        state_emb = self._slot_embedding(self.state_embedding, states_info, state_channels)

        if self.context_only:
            # no lookup, so task ids of envs added after training are accepted
            task_emb = states.new_zeros(states.shape[0], 1, self.task_embedding.embedding_dim + self.model_params_dim)
        else:
            task_emb = torch.cat([
                self.task_embedding(task_ids).unsqueeze(1),  # [bs, 1, d_model]
                model_params.unsqueeze(1)  # [bs, 1, num_params]
            ], dim=2)  # [bs, 1, d_model + num_params]

        action_emb = self._slot_embedding(self.action_embedding, actions_info, action_channels)
        reward_emb = self.reward_embedding(reward_input)

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

        actions_pred = prev_actions + self.action_head(encoded)  # [bs, seq_length, action_dim]

        # Optional PINN predictions
        pinn_pred = None
        if self.has_pinn:
            pinn_pred = self.pinn_head(encoded) # [bs, seq_length-1, pinn_output_dim]

        return actions_pred, pinn_pred

    def _get_state_info(self, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        state_values, state_ids = [], []
        for state_name, state_value in state.items():
            state_ids.append(self.tokenizer.state_token_id(state_name))
            state_values.append(_to_scalar(state_value))

        state_values += [0] * (self.state_dim - len(state_values))
        empty_token_id = self.tokenizer.state_mapping["Empty"]
        state_ids += [empty_token_id] * (self.state_dim - len(state_ids))
        return torch.tensor(state_values, dtype=torch.float32), torch.tensor(state_ids, dtype=torch.long)

    def _get_action_info(self, action: dict) -> tuple[torch.Tensor, torch.Tensor]:
        action_values, action_ids = [], []
        for action_name, action_value in action.items():
            action_ids.append(self.tokenizer.action_token_id(action_name))
            action_values.append(_to_scalar(action_value))
        action_values += [0] * (self.action_dim - len(action_values))
        empty_token_id = self.tokenizer.action_mapping["Empty"]
        action_ids += [empty_token_id] * (self.action_dim - len(action_ids))
        return torch.tensor(action_values, dtype=torch.float32), torch.tensor(action_ids, dtype=torch.long)

    @staticmethod
    def _clip_to_action_space(env: AbstractEconomicEnv, values: np.ndarray) -> np.ndarray:
        """Clip actions to the env's Box bounds where declared."""
        space = getattr(env, "action_space", None)
        if isinstance(space, gym.spaces.Box):
            low, high = space.low.reshape(-1), space.high.reshape(-1)
            n = min(len(values), len(low))
            values[:n] = np.clip(values[:n], low[:n], high[:n])
        return values

    @staticmethod
    def _env_step(env: AbstractEconomicEnv, values: np.ndarray):
        """Call env.step as the env expects: one kwarg per action, a dict per agent, or an array
        (a scalar for single-action envs)."""
        names = list(env.action_description)
        if list(inspect.signature(env.step).parameters) == names:
            return env.step(**{name: np.float64(v) for name, v in zip(names, values)})
        if hasattr(env, "agent_ids"):
            return env.step({agent: values.copy() for agent in env.agent_ids})
        return env.step(values if len(values) > 1 else np.float64(values[0]))

    @staticmethod
    def _scalar_reward(reward) -> float:
        # multi-agent envs: total over agents
        if isinstance(reward, dict):
            return float(sum(_to_scalar(v) for v in reward.values()))
        return _to_scalar(reward)

    def inference(self, env: AbstractEconomicEnv, max_steps: int = 50) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        state_names, action_names = list(env.state_description), list(env.action_description)
        if len(state_names) > self.state_dim or len(action_names) > self.action_dim:
            raise ValueError(
                f"{type(env).__name__} has {len(state_names)} states / {len(action_names)} actions, but the "
                f"model was built with state_dim={self.state_dim} / action_dim={self.action_dim} "
                "(train.max_state_dim / train.max_action_dim)."
            )
        device = next(self.parameters()).device

        init_state, _ = env.reset()
        state = {name: _to_scalar(init_state[name]) for name in state_names}
        action = {name: 0.0 for name in action_names}  # a_{-1} = 0, as in training
        _, states_info = self._get_state_info(state)
        _, actions_info = self._get_action_info(action)

        state_to_plot, action_to_plot = [state], [action]
        state_history = [self._get_state_info(state)[0]]
        action_history = [self._get_action_info(action)[0]]
        reward_history = [torch.tensor([0.0], dtype=torch.float32)]
        task_ids = torch.tensor([env.task_id], dtype=torch.long)
        numeric_params = [v for _, v in sorted(env.params.items()) if isinstance(v, (int, float))][:self.model_params_dim]
        model_params = torch.tensor(numeric_params + [0.0] * (self.model_params_dim - len(numeric_params)), dtype=torch.float32)

        for _ in range(max_steps):
            window = slice(-self.max_seq_len, None)
            out, _ = self.forward(
                states=torch.stack(state_history[window]).unsqueeze(0).to(device),
                states_info=states_info.unsqueeze(0).to(device),
                actions=torch.stack(action_history[window]).unsqueeze(0).to(device),
                actions_info=actions_info.unsqueeze(0).to(device),
                rewards=torch.stack(reward_history[window]).unsqueeze(0).to(device),
                task_ids=task_ids.to(device),
                model_params=model_params.unsqueeze(0).to(device),
            )
            predicted = out[0, -1, :len(action_names)].detach().cpu().numpy().astype(np.float64)
            executed = self._clip_to_action_space(env, predicted)
            next_state, reward, _, _, _ = self._env_step(env, executed)

            state = {name: _to_scalar(next_state[name]) for name in state_names}
            action = {name: float(v) for name, v in zip(action_names, executed)}
            state_to_plot.append(state)
            action_to_plot.append(action)
            state_history.append(self._get_state_info(state)[0])
            action_history.append(self._get_action_info(action)[0])
            reward_history.append(torch.tensor([self._scalar_reward(reward)], dtype=torch.float32))

        return state_to_plot, action_to_plot
