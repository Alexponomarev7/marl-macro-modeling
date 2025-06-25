import warnings; warnings.filterwarnings("ignore")

import numpy as np
import gymnasium as gym

from lib.envs.environment_base import ENV_TO_ID, AbstractEconomicEnv


class RamseyEnv(AbstractEconomicEnv):
    """
    Ramsey growth model environment with CRRA utility (log-utility for CRRA=1)
    """

    def __init__(
            self,
            alpha: float = 0.33,
            beta: float = 0.96,
            delta: float = 0.1,
            start_capital: float = 1.0,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.start_capital = start_capital

        # Steady state capital and consumption
        self.k_ss = ((1 / beta - (1 - delta)) / alpha) ** (1 / (alpha - 1))
        self.c_ss = self.k_ss ** alpha - delta * self.k_ss

        # Spaces
        self.observation_space = gym.spaces.Dict({
            "Capital": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "Output": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "Consumption": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "Investment": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        })

        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([np.inf, np.inf]),
            dtype=np.float32,
        )


    @staticmethod
    def name() -> str:
        return "Ramsey"

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset environment with dynamically selected c0"""
        super().reset(seed=seed)
        self.current_step = 0
        self.capital: float = self.start_capital
        self.output = 0
        # self.step(self.find_optimal_c0())
        # self.output = 0
        self.consumption = 0
        self.investment = 0
        return self._get_state(), {}

    def step(self, Consumption: float) -> tuple[dict, float, bool, bool, dict]:
        """Advance model"""
        self.output = self.capital ** self.alpha
        self.consumption = Consumption
        self.investment = self.output - self.consumption
        self.capital = self.capital * (1 - self.delta) + self.investment

        reward = np.log(self.consumption) if self.consumption > 0 else -np.inf
        self.current_step += 1
        done = False

        info = {
            "consumption": self.consumption,
            "investment": self.investment,
            "utility": reward,
        }

        return self._get_state(), reward, done, False, info

    def _get_state(self) -> dict:
        return {
            "Consumption": self.consumption,
            "Capital": self.capital,
            "Output": self.output,
            "Investment": self.investment,
        }

    def simulate_trajectory(self, c0_guess: float, T: int = 100) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        k_path = np.zeros(T + 1)
        c_path = np.zeros(T)
        k_path[0] = self.start_capital
        c_path[0] = c0_guess

        for t in range(T):
            y = k_path[t] ** self.alpha
            i = y - c_path[t]
            k_next = k_path[t] * (1 - self.delta) + i
            if k_next <= 0 or c_path[t] <= 0:
                return None, None
            k_path[t + 1] = k_next
            if t + 1 < T:
                c_path[t + 1] = c_path[t] * (
                    self.beta * (self.alpha * k_path[t + 1] ** (self.alpha - 1) + 1 - self.delta)
                )

        return k_path, c_path

    def find_optimal_c0(self, c_min: float = 0.01, c_max: float = 2.0, tol: float = 1e-8, T: int = 100) -> float:
        """Binary search for optimal initial consumption"""
        for _ in range(100):
            c_try = (c_min + c_max) / 2
            k_path, _ = self.simulate_trajectory(c_try, T)
            if k_path is None:
                c_max = c_try
                continue
            if abs(k_path[-1] - self.k_ss) < tol:
                return c_try
            if k_path[-1] > self.k_ss:
                c_min = c_try
            else:
                c_max = c_try
        return (c_min + c_max) / 2

    def analytical_step(self) -> tuple[dict, float, bool, bool, dict]:
        """
        Advance the model one step using the Euler equation:
        1/C = beta * 1/C(+1) * (alpha*Y(+1)/K + 1-delta)
        """
        if self.current_step == 0:
            next_consumption = self.find_optimal_c0()
        else:
            next_output = self.capital ** self.alpha
            next_consumption = self.consumption * self.beta * (self.alpha * next_output / self.capital + 1 - self.delta)
        return self.step(next_consumption)

    def render(self):
        return super().render()

    def close(self):
        return super().close()

    @property
    def task_id(self) -> int:
        return ENV_TO_ID["Ramsey"]

    @property
    def params(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "delta": self.delta,
            "start_capital": self.start_capital
        }

    @property
    def state_description(self) -> dict[str, str]:
        return {
            "Capital": "Capital stock",
            "Output": "Current output",
        }

    @property
    def action_description(self) -> dict[str, str]:
        return {
            "Consumption": "Consumption",
        }
