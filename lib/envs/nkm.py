from typing import Dict, Tuple, Union

import numpy as np
import gymnasium as gym

from lib.envs.environment_base import AbstractEconomicEnv


class NKMEnv(AbstractEconomicEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.observation_space = gym.spaces.Dict(
            {
                "Inflation": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "Output": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )

        self.action_space = gym.spaces.Box(
            low=np.array([0]),
            high=np.array([1]),
            dtype=np.float32,
        )

        # self._map_action_to_name = {
        #     0: "leisure",
        #     1: "consumption",
        #     2: "investment",
        # }

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[Dict, Dict]:
        super().reset(seed, options)

        self.current_step = 0
        self.capital = self.initial_capital
        self.technology = 0.0  # Initial technology level (log)
        self.labor = 0.5  # Initial labor supply
        self.output = self._calculate_output()

        return self._get_state(), {}

    def _get_state(self) -> Dict:
        """
        Construct the current state observation dictionary.

        :return: Dictionary containing current values of all state variables
        """
        return {
            "Inflation": np.array([self.inflation], dtype=np.float32),
            "Output": np.array([self.output], dtype=np.float32),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict | float | bool]:
        """
        Execute one time step of the RBC model.

        :param action: Array of [inflation, output, consumption_rate]
                      Each component must be between 0 and 1
        :return: Tuple containing:
                - state: Dictionary of current state variables
                - reward: Utility value for current period
                - terminated: Whether episode has ended (always False)
                - truncated: Whether episode was truncated (always False)
                - info: Dictionary with additional information
        """





    def render(self):
        return super().render()

    def close(self):
        return super().close()

    def analytical_step(self) -> Tuple[Dict | float | bool]:
        return super().analytical_step()

    @property
    def params(self) -> Dict[str, Union[float, str, dict]]:
        """
        Get the current parameters of the economic environment.

        :return: Dictionary containing the current parameters of the environment
        """
        pass

    @property
    def state_description(self):
        """
        Provide descriptions of state variables.

        :return: Dictionary mapping state variable names to their descriptions
        """
        pass

    @property
    def action_description(self):
        """
        Provide descriptions of action variables.

        :return: Dictionary mapping action variable names to their descriptions
        """
        pass
