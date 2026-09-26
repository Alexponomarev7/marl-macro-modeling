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


UNIT_LEVEL_FLOOR = 1e-3  # share of the running level added to the causal-mode unit of change


def _running_stats(x: torch.Tensor, level_ok: torch.Tensor, delta_ok: torch.Tensor):
    """Running mean and std of the level and RMS of the first differences over positions <= t.

    x: [batch, seq, dim]; level_ok / delta_ok: [batch, seq, 1] masks of observed values / differences.
    """
    dtype = torch.float32 if x.device.type == "mps" else torch.float64
    x = x.to(dtype)
    lv, dv = level_ok.to(dtype), delta_ok.to(dtype)
    # shift by the first observed value to avoid cancellation in E[x^2] - E[x]^2
    first = level_ok.long().argmax(dim=1, keepdim=True).expand(-1, 1, x.shape[-1])
    ref = torch.gather(x, 1, first)
    y = (x - ref) * lv
    count = torch.cumsum(lv, 1)
    mean_y = torch.cumsum(y, 1) / count.clamp(min=1)
    std = (torch.cumsum(y * y, 1) / count.clamp(min=1) - mean_y ** 2).clamp(min=0).sqrt()
    mean = torch.where(count > 0, ref + mean_y, 0.0)
    dx = _first_difference(x) * dv
    rms = (torch.cumsum(dx * dx, 1) / torch.cumsum(dv, 1).clamp(min=1)).sqrt()
    return mean, std, rms, dx


def causal_features(x: torch.Tensor, level_ok: torch.Tensor, delta_ok: torch.Tensor, lagged: bool = False):
    """Scale-free channels of each variable from positions <= t: the level's running z-score, the
    first difference over the running RMS of differences and, with `lagged`, the previous difference.

    Returns channels [batch, seq, dim, C] and the unit of change [batch, seq, dim].
    """
    mean, std, rms, dx = _running_stats(x, level_ok, delta_ok)
    eps = 1e-6 * mean.abs() + 1e-12
    channels = [torch.where(level_ok, (x.to(mean.dtype) - mean) / (std + eps), 0.0), dx / (rms + eps)]
    if lagged:
        channels.append(torch.cat([torch.zeros_like(dx[:, :1]), dx[:, :-1]], 1) / (rms + eps))
    unit = rms + UNIT_LEVEL_FLOOR * mean.abs() + 1e-12
    return torch.stack(channels, -1).clamp(-10.0, 10.0).float(), unit.float()


RIDGE_PENALTY = 1e-4  # on features normalized by their sum of squares over the context
RIDGE_MIN_PAIRS = 3
RIDGE_MOVE_FLOOR = 1e-10  # a state whose changes in the context have a smaller root sum of squares is constant


def in_context_ridge_change(states: torch.Tensor, prev_actions: torch.Tensor, state_delta_ok: torch.Tensor,
                            action_delta_ok: torch.Tensor, window: int | None = None,
                            penalty: float = RIDGE_PENALTY) -> torch.Tensor:
    """In-context ridge estimate of a_t - a_{t-1}: regress a_tau - a_{tau-1} on s_tau - s_{tau-1}
    over tau < t (the last `window` pairs if given) and evaluate at s_t - s_{t-1}; 0 until there
    are RIDGE_MIN_PAIRS pairs. The pair of step tau is complete at position tau + 1.

    states [B, L, S], prev_actions [B, L, A]; *_delta_ok [B, L, 1]. Returns [B, L, A].
    Runs on the CPU: batched linalg.solve is slow on MPS.
    """
    device, dtype = prev_actions.device, prev_actions.dtype
    states, prev_actions = states.detach().cpu().double(), prev_actions.detach().cpu().double()
    state_delta_ok, action_delta_ok = state_delta_ok.cpu(), action_delta_ok.cpu()
    ds = _first_difference(states) * state_delta_ok        # s_t - s_{t-1} at t
    da = _first_difference(prev_actions) * action_delta_ok  # a_{t-1} - a_{t-2} at t
    lag = lambda z: torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], 1)
    pair_ok = (action_delta_ok & lag(state_delta_ok)).double()       # pair of step t-1, complete at t
    x, y = lag(ds) * pair_ok, da * pair_ok
    total = lambda z: torch.cumsum(z, 1)
    if window:  # pairs completed in (t - window, t]
        total = lambda z: torch.cumsum(z, 1) - torch.cat([torch.zeros_like(z[:, :window]), torch.cumsum(z, 1)[:, :-window]], 1)
    xtx = total(x.unsqueeze(-1) * x.unsqueeze(-2))                    # [B, L, S, S], pairs completed by t
    xty = total(x.unsqueeze(-1) * y.unsqueeze(-2))                    # [B, L, S, A]
    scale = torch.diagonal(xtx, dim1=-2, dim2=-1).sqrt()               # [B, L, S]
    moving = scale > RIDGE_MOVE_FLOOR
    scale = torch.where(moving, scale, 1.0)
    gram = xtx / (scale.unsqueeze(-1) * scale.unsqueeze(-2)) + penalty * torch.eye(states.shape[-1], dtype=torch.float64)
    coef = torch.linalg.solve(gram, xty / scale.unsqueeze(-1))        # standardized G_hat
    change = torch.einsum("bls,blsa->bla", torch.where(moving, ds / scale, 0.0), coef)
    enough = total(pair_ok) >= RIDGE_MIN_PAIRS
    return torch.where(enough, change, 0.0).to(device=device, dtype=dtype)


def causal_zscore(x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Running z-score of x [batch, seq, dim], skipping positions where valid [batch, seq] is False."""
    ok = torch.ones_like(x[..., :1], dtype=torch.bool) if valid is None else valid.bool().unsqueeze(-1)
    return causal_features(x, ok, ok)[0][..., 0]


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
        input_normalization: str = "symlog",
        ridge_channel: bool = False,
        ridge_windows: tuple[int, ...] = (),
        ridge_skip: bool = False,
        ridge_penalty: float = RIDGE_PENALTY,
    ):
        """context_only: hide the task id and the model parameters.
        input_normalization: "symlog" or "causal" (scale-free running statistics, see causal_features).
        ridge_channel: add in_context_ridge_change to each action slot (causal mode only).
        ridge_windows: also add the estimate over each of these last numbers of steps.
        ridge_skip: predict a_{t-1} plus a learned convex mix of no change and the ridge estimates,
            plus a gated correction from the head.
        """
        super().__init__()
        if input_normalization not in ("symlog", "causal"):
            raise ValueError(f"input_normalization must be 'symlog' or 'causal', got {input_normalization!r}")
        if ridge_channel and input_normalization != "causal":
            raise ValueError("ridge_channel needs input_normalization='causal'")
        if (ridge_windows or ridge_skip) and not ridge_channel:
            raise ValueError("ridge_windows and ridge_skip need ridge_channel")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len
        self.d_model = d_model * (1 + state_dim + action_dim + 1)
        self.has_pinn = has_pinn
        self.pinn_output_dim = pinn_output_dim
        self.model_params_dim = model_params_dim
        self.context_only = context_only
        self.input_normalization = input_normalization
        self.ridge_channel = ridge_channel
        self.ridge_windows = tuple(ridge_windows)
        self.ridge_skip = ridge_skip
        self.ridge_penalty = ridge_penalty

        self.tokenizer = Tokenizer()
        # slot = [variable-name embedding | numeric channels]
        state_channels = 3 if input_normalization == "causal" else 2
        self.state_embedding = nn.Embedding(self.tokenizer.num_state_tokens, d_model - state_channels)
        self.action_embedding = nn.Embedding(self.tokenizer.num_action_tokens, d_model - 2 - ridge_channel - len(self.ridge_windows))
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
        if ridge_skip:
            # per action: logits of no change and of each ridge estimate, then the correction's gate
            self.ridge_gate = nn.Linear(self.d_model, action_dim * (3 + len(self.ridge_windows)), dtype=torch.float32)
            nn.init.zeros_(self.ridge_gate.weight)
            nn.init.zeros_(self.ridge_gate.bias)
            with torch.no_grad():
                self.ridge_gate.bias.view(action_dim, -1)[:, 1] = 4.0  # start near the full-window estimate

        # Optional PINN head for predicting additional data
        if self.has_pinn:
            self.pinn_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, pinn_output_dim)
            )

    @staticmethod
    def _observed_steps(states: torch.Tensor, first_step: torch.Tensor | None, attention_mask: torch.Tensor | None):
        """[batch, seq, 1] masks of real s_t and of real (a_{t-1}, r_{t-1}): padding is neither, and
        an episode's first step has placeholder a_{-1}, r_{-1}."""
        batch, seq = states.shape[:2]
        valid = torch.ones(batch, seq, 1, dtype=torch.bool, device=states.device)
        if attention_mask is not None:
            valid = attention_mask.bool().view(batch, seq, 1)
        if first_step is None:
            return valid, valid
        first_valid = valid & ~torch.cat([torch.zeros_like(valid[:, :1]), valid[:, :-1]], 1)
        return valid, valid & ~(first_valid & first_step.bool().view(batch, 1, 1))

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
        first_step: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass creating sequences of states, actions, and rewards, and predicts actions for each timestep.

        Args:
            states: [batch_size, seq_length, state_dim] - s_t
            actions: [batch_size, seq_length, action_dim] - a_{t-1} (0 at an episode's first step)
            rewards: [batch_size, seq_length, 1] - r_{t-1} (0 at an episode's first step)
            task_ids: [batch_size]
            first_step: [batch_size] bool, the window starts the episode (causal mode only)
            attention_mask: [batch_size, seq_length] bool, non-padding positions (causal mode only)

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Predicted actions and optional PINN predictions
        """
        seq_length = states.shape[1]
        bound = 1e9 if self.input_normalization == "causal" else 1000.0
        states, actions, rewards, model_params = (
            torch.nan_to_num(x, nan=0.0, posinf=bound, neginf=-bound).clamp(-bound, bound)
            for x in (states, actions, rewards, model_params)
        )
        prev_actions = actions  # step t carries a_{t-1}
        if self.input_normalization == "causal":
            valid, prev_valid = self._observed_steps(states, first_step, attention_mask)
            after = lambda ok: ok & torch.cat([torch.zeros_like(ok[:, :1]), ok[:, :-1]], 1)  # at t and t-1
            state_channels, _ = causal_features(states, valid, after(valid), lagged=True)
            action_channels, action_unit = causal_features(prev_actions, prev_valid, after(prev_valid))
            if self.ridge_channel:
                ridges = [(in_context_ridge_change(states, prev_actions, after(valid), after(prev_valid), window, self.ridge_penalty)
                           / action_unit).clamp(-10.0, 10.0).float() for window in (None, *self.ridge_windows)]  # in action units
                action_channels = torch.cat([action_channels] + [r.unsqueeze(-1) for r in ridges], -1)
            reward_channels, _ = causal_features(rewards, prev_valid, after(prev_valid))
            reward_input = reward_channels[..., 0]  # level z-score
        else:
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

        residual = self.action_head(encoded)
        if self.ridge_skip:
            gates = self.ridge_gate(encoded).view(*residual.shape, -1)
            options = torch.stack([torch.zeros_like(residual)] + ridges, -1)
            residual = (gates[..., :-1].softmax(-1) * options).sum(-1) + torch.sigmoid(gates[..., -1]) * residual
        if self.input_normalization == "causal":
            residual = residual * action_unit
        actions_pred = prev_actions + residual  # [bs, seq_length, action_dim]

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

    def inference(
        self, env: AbstractEconomicEnv, max_steps: int = 50, initial_action: dict[str, float] | None = None,
    ) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        """Roll the policy out in env; initial_action (required in causal mode) is the first action."""
        state_names, action_names = list(env.state_description), list(env.action_description)
        if len(state_names) > self.state_dim or len(action_names) > self.action_dim:
            raise ValueError(
                f"{type(env).__name__} has {len(state_names)} states / {len(action_names)} actions, but the "
                f"model was built with state_dim={self.state_dim} / action_dim={self.action_dim} "
                "(train.max_state_dim / train.max_action_dim)."
            )
        if self.input_normalization == "causal" and initial_action is None:
            raise ValueError("a causal-mode policy needs initial_action")
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

        for step in range(max_steps):
            if step == 0 and initial_action is not None:
                predicted = np.array([initial_action[name] for name in action_names], dtype=np.float64)
            else:
                window = slice(-self.max_seq_len, None)
                out, _ = self.forward(
                    states=torch.stack(state_history[window]).unsqueeze(0).to(device),
                    states_info=states_info.unsqueeze(0).to(device),
                    actions=torch.stack(action_history[window]).unsqueeze(0).to(device),
                    actions_info=actions_info.unsqueeze(0).to(device),
                    rewards=torch.stack(reward_history[window]).unsqueeze(0).to(device),
                    task_ids=task_ids.to(device),
                    model_params=model_params.unsqueeze(0).to(device),
                    first_step=torch.tensor([len(state_history) <= self.max_seq_len], device=device),
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
