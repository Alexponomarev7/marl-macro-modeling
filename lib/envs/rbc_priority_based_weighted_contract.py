import warnings

warnings.filterwarnings("ignore")

import numpy as np
import gymnasium as gym
from typing import Optional, Dict, Tuple, Union
from lib.envs.environment_base import AbstractEconomicEnv
from lib.utility_funcs import log_utility, ces_utility

class RBCPriorityBasedWeightedContractEnv(AbstractEconomicEnv):
    """
    A Real Business Cycle (RBC) environment implementing a standard RBC model.
    The agent acts as a representative consumer-worker making decisions about
    consumption, investment, and labor supply.

    State Space:
        - Capital: Current capital stock
        - Labor: Labor supply (1 - leisure)
        - Technology: Technology level (in logs)
        - Output: Current period output

    Action Space:
        - Investment rate: Fraction of output for investment
        - Leisure: Time allocated to leisure
        - Consumption rate: Fraction of output for consumption

    :param discount_rate: Time preference parameter (discount factor in RL), defaults to 0.99
    :param marginal_disutility_of_labor: Weight on leisure in utility, defaults to 1.0
    :param depreciation_rate: Capital depreciation rate, defaults to 0.025
    :param capital_share_of_output: Capital share in production function, defaults to 0.36
    :param technology_shock_persistence: AR(1) coefficient for technology, defaults to 0.95
    :param technology_shock_variance: Variance of technology innovations, defaults to 0.007
    :param initial_capital: Starting capital stock, defaults to 1.0
    :param max_capital: Upper bound on capital stock, defaults to 10.0
    :param utility_function: Type of utility function ('log' or 'ces'), defaults to 'log'
    :param utility_params: Dictionary of utility function parameters
    """
    def __init__(
        self,
        discount_rate: float = 0.99,
        marginal_disutility_of_labor: float = 1.0,
        depreciation_rate: float = 0.025,
        capital_share_of_output: float = 0.36,
        technology_shock_persistence: float = 0.95,
        technology_shock_variance: float = 0.007,
        initial_capital: float = 1.0,
        max_capital: float = 10.0,
        utility_function: str = "log",
        utility_params: dict = None,
        agent_weights: Dict[str, float] = None,
    ):
        super().__init__()

        self.discount_rate = discount_rate
        self.marginal_disutility_of_labor = marginal_disutility_of_labor
        self.depreciation_rate = depreciation_rate
        self.capital_share_of_output = capital_share_of_output
        self.technology_shock_persistence = technology_shock_persistence
        self.technology_shock_variance = technology_shock_variance
        self.initial_capital = initial_capital
        self.max_capital = max_capital

        self._set_utility_function(utility_function, utility_params or {})

        self.agent_weights = agent_weights or {"agent_1": 1.0, "agent_2": 1.0}

        self.observation_space = gym.spaces.Dict(
            {
                "Capital": gym.spaces.Box(low=0.0, high=max_capital, shape=(1,), dtype=np.float32),
                "Labor": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "Technology": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "Output": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )

        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float32,
        )

    def _set_utility_function(self, utility_function: str, utility_params: dict):
        self.utility_params = utility_params

        if utility_function.lower() == "log":
            self.utility_function = log_utility
            self.utility_params.setdefault('A', self.marginal_disutility_of_labor)

        elif utility_function.lower() == "ces":
            self.utility_function = ces_utility
            self.utility_params.setdefault('sigma', 2.0)
            self.utility_params.setdefault('eta', 1.5)
            self.utility_params.setdefault('A', self.marginal_disutility_of_labor)

        else:
            raise ValueError(f"Unknown utility function: {utility_function}. Choose 'log' or 'ces'")

    def calculate_utility(
        self, consumption: Union[float, np.ndarray], labor: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return self.utility_function(C=consumption, L=labor, **self.utility_params)

    def _split_rewards(self, successes: Dict[str, bool]) -> Dict[str, float]:
        """
        Distribute rewards among agents based on priority-based weighted contracts.

        :param success: Boolean indicating if the main agent succeeded in improving its state
        :return: Dictionary mapping agent IDs to their rewards
        """
        total_weight = sum(
            self.agent_weights[agent] for agent, success in successes.items() if success
        )
        rewards = {}
        for agent, success in successes.items():
            if success:
                rewards[agent] = self.agent_weights[agent] / total_weight
            else:
                rewards[agent] = 0.0
        return rewards

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict[str, float], bool, bool, Dict]:
        """
        Execute one time step of the RBC model, updating the state and distributing rewards
        among agents based on priority-based weighted contracts.

        :param action: Array of [investment_rate, leisure, consumption_rate]
                      Each component must be between 0 and 1
        :return: Tuple containing:
                - state: Dictionary of current state variables
                - reward: Utility value for current period
                - terminated: Whether episode has ended (always False)
                - truncated: Whether episode was truncated (always False)
                - info: Dictionary with additional information
        """
        rewards = {}
        successes = {}

        for agent, action in actions.items():
            leisure = np.clip(action[1], 0, 1)
            investment_consumption = np.clip(action[[0, 2]], 0, 1)
            investment_consumption = investment_consumption / np.sum(investment_consumption)

            investment_rate, consumption_rate = investment_consumption
            new_labor = 1 - leisure

            technology_shock = np.random.normal(0, self.technology_shock_variance)
            new_technology = (
                self.technology_shock_persistence * self.technology + technology_shock
            )

            current_output = self._calculate_output()
            investment = investment_rate * current_output
            consumption = consumption_rate * current_output
            new_capital = (1 - self.depreciation_rate) * self.capital + investment
            new_capital = np.clip(new_capital, 0, self.max_capital)

            self.capital = new_capital
            self.labor = new_labor
            self.technology = new_technology
            self.output = self._calculate_output()

            reward = self.calculate_utility(consumption, new_labor)
            rewards[agent] = reward
            successes[agent] = reward > 0  # Define success criteria

        reward_distribution = self._split_rewards(successes)
        return self._get_state(), reward_distribution, False, False, {}

    def _calculate_output(self) -> float:
        return np.exp(self.technology) * (self.capital ** self.capital_share_of_output) * \
               (self.labor ** (1 - self.capital_share_of_output))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        self.current_step = 0
        self.capital = self.initial_capital
        self.technology = 0.0
        self.labor = 0.5
        self.output = self._calculate_output()

        return self._get_state(), {}

    def _get_state(self) -> Dict:
        return {
            "Capital": np.array([self.capital], dtype=np.float32),
            "Labor": np.array([self.labor], dtype=np.float32),
            "Technology": np.array([self.technology], dtype=np.float32),
            "Output": np.array([self.output], dtype=np.float32),
        }

    def render(self):
        state = self._get_state()
        print("\nCurrent State:")
        for key, value in state.items():
            print(f"{key}: {value[0]:.4f}")

    def close(self):
        """Clean up resources"""
        # included for compatibility with the Gymnasium API
        pass

    @property
    def params(self) -> Dict[str, Union[float, str, dict]]:
        return {
            "discount_rate": self.discount_rate,
            "marginal_disutility_of_labor": self.marginal_disutility_of_labor,
            "depreciation_rate": self.depreciation_rate,
            "capital_share_of_output": self.capital_share_of_output,
            "technology_shock_persistence": self.technology_shock_persistence,
            "technology_shock_variance": self.technology_shock_variance,
            "initial_capital": self.initial_capital,
            "max_capital": self.max_capital,
            "utility_function": self.utility_function.__name__,
            "utility_params": self.utility_params,
        }

    @property
    def state_description(self):
        return {
            "Capital": "Capital stock",
            "Labor": "Labor supply = (1 - leisure)",
            "Technology": "Technology level (log)",
            "Output": "Current output"
        }

    @property
    def action_description(self):
        return {
            "investment_rate": "Fraction of output allocated to investment",
            "leisure": "Time allocated to leisure",
            "consumption_rate": "Fraction of output allocated to consumption"
        }