import warnings; warnings.filterwarnings("ignore")

import numpy as np
import gymnasium as gym

from lib.envs.environment_base import ENV_TO_ID, AbstractEconomicEnv


class RamseyEnv(AbstractEconomicEnv):
    """
    :param alpha: Capital share in production function, defaults to 0.33
    :param beta: Discount factor, defaults to 0.96
    :param delta: Capital depreciation rate, defaults to 0.1
    :param start_capital: Initial capital stock, defaults to 1.0
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

        # Calculate steady state values
        self.k_ss = ((1/beta - (1 - delta))/alpha)**(1/(alpha - 1))
        self.c_ss = self.k_ss**alpha - delta*self.k_ss

        # Define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "Capital": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "Output": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "Consumption": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "Investment": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )

        # Define action space for consumption and investment rates
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([np.inf, np.inf]),
            dtype=np.float32,
        )

    def analytical_step(self) -> dict | float | bool | bool | dict:
        """Analytical step"""
        raise NotImplementedError("Analytical step not implemented")
    
    def render(self):
        return super().render()
    
    def close(self):
        return super().close()

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.capital = self.start_capital
        self.output = self.capital**self.alpha
        self.consumption = self.c_ss
        self.investment = self.output - self.consumption
        return self._get_state(), {}

    def _get_state(self) -> dict:
        """Get current state observation"""
        return {
            "Capital": np.array([self.capital], dtype=np.float32),
            "Output": np.array([self.output], dtype=np.float32),
            "Consumption": np.array([self.consumption], dtype=np.float32),
            "Investment": np.array([self.investment], dtype=np.float32),
        }
    
    def step(self, consumption: float) -> tuple[dict, float, bool, bool, dict]:
        """Execute one timestep"""
        # Calculate consumption and investment
        self.consumption = consumption
        self.investment = self.output - self.consumption

        # Update capital and output
        self.capital = (1 - self.delta) * self.capital + self.investment
        self.output = self.capital**self.alpha

        # Calculate reward (utility)
        reward = np.log(self.consumption) if self.consumption > 0 else -np.inf

        # Update step counter
        self.current_step += 1
        done = False

        info = {
            "consumption": self.consumption,
            "investment": self.investment,
            "utility": reward
        }

        return self._get_state(), reward, done, False, info

    @property
    def task_id(self) -> int:
        return ENV_TO_ID["Ramsey"]

    @property
    def params(self) -> dict[str, float]:
        """Get environment parameters"""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "delta": self.delta,
            "start_capital": self.start_capital
        }

    @property
    def state_description(self) -> dict[str, str]:
        """Get state variable descriptions"""
        return {
            "Capital": "Capital stock",
            "Output": "Current output",
            "Consumption": "Current consumption",
            "Investment": "Current investment"
        }

    @property
    def action_description(self) -> dict[str, str]:
        """Get action variable descriptions"""
        return {
            "consumption_rate": "Fraction of output for consumption",
            "investment_rate": "Fraction of output for investment"
        }
