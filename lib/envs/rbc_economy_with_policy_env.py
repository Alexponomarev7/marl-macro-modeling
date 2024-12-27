import warnings; warnings.filterwarnings("ignore")

import numpy as np
import gymnasium as gym
from typing import (
    Optional,
    Dict,
    Tuple,
)

from lib.envs.environment_base import AbstractEconomicEnv
from lib.utility_funcs import (
    log_utility,
    ces_utility,
)

class RBCEconomyWithPolicyEnv(AbstractEconomicEnv):
    """
    Extention of Real Business Cycle (RBC) model with fiscal and monetary policy mechanisms.
    It simulates a dynamic economic environment where the state evolves based on policy decisions, labor supply, and technology shocks.
    Fiscal and Monetary Policy:
   - The environment allows for the adjustment of tax rates, government spending, and the money supply, reflecting real-world policy levers.
   - Actions include changes to these variables, influencing the overall economic trajectory.
    Dynamic State Variables:
   - The state includes `Capital`, `Technology`, `TaxRate`, `GovSpending`, and `MoneySupply`, among others, representing the key components of an economy.
   - These variables evolve through the simulation based on economic dynamics and the policy choices made.
    Technology Shocks:
   - A persistent stochastic process governs technology (`Technology`), introducing randomness and mimicking real-world uncertainty in productivity.
   - Economic output depends on labor and capital using a Cobb-Douglas production function, modified by technology.
    Action and Observation Spaces:
   - Actions include rates for investment, leisure, and adjustments to policy levers (tax rate, government spending, and money supply).
   - Observations describe the current state of the economy, bounded appropriately to reflect real-world constraints.
   - Rewards are tied to utility, promoting policies that balance consumption and labor while ensuring sustainable investment and capital growth.
    Constraints and Bounds:
   - Realistic constraints, such as maximum capital stock and non-negative government spending, are enforced to prevent implausible economic states.
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
        initial_tax_rate: float = 0.2,
        initial_gov_spending: float = 0.2,
        initial_money_supply: float = 1.0,
        max_capital: float = 10.0,
        utility_function: str = "log",
        utility_params: dict = None,
    ):
        super().__init__()

        self.discount_rate = discount_rate
        self.marginal_disutility_of_labor = marginal_disutility_of_labor
        self.depreciation_rate = depreciation_rate
        self.capital_share_of_output = capital_share_of_output
        self.technology_shock_persistence = technology_shock_persistence
        self.technology_shock_variance = technology_shock_variance

        self.capital = initial_capital
        self.tax_rate = initial_tax_rate
        self.gov_spending = initial_gov_spending
        self.money_supply = initial_money_supply
        self.technology = 0.0

        self.max_capital = max_capital

        # Utility function setup
        self._set_utility_function(utility_function, utility_params or {})

        self.observation_space = gym.spaces.Dict({
            "Capital": gym.spaces.Box(low=0.0, high=max_capital, shape=(1,), dtype=np.float32),
            "Labor": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "Technology": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "Output": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "TaxRate": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "GovSpending": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "MoneySupply": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        })

        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -0.1, -1.0, -0.5]),
            high=np.array([1.0, 1.0, 0.1, 1.0, 0.5]),
            dtype=np.float32,
        )

    def _set_utility_function(self, utility_function: str, utility_params: dict):
        if utility_function.lower() == "log":
            self.utility_function = log_utility
            self.utility_params = utility_params
        elif utility_function.lower() == "ces":
            self.utility_function = ces_utility
            self.utility_params = utility_params
        else:
            raise ValueError(f"Unknown utility function: {utility_function}")

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        investment_rate, leisure, tax_rate_change, gov_spending_change, money_supply_change = action

        # Update fiscal and monetary policy variables
        self.tax_rate = np.clip(self.tax_rate + tax_rate_change, 0.0, 1.0)
        self.gov_spending = max(self.gov_spending + gov_spending_change, 0.0)
        self.money_supply = max(self.money_supply + money_supply_change, 0.0)

        labor_supply = 1 - leisure

        # Technology shock
        self.technology += (
            self.technology_shock_persistence * self.technology
            + np.random.normal(0, self.technology_shock_variance)
        )

        # Output calculation
        output = self._calculate_output(labor_supply)

        net_output = output * (1 - self.tax_rate)
        gov_investment = self.tax_rate * output

        # Investment and consumption
        investment = investment_rate * net_output
        consumption = (1 - investment_rate) * net_output

        # Update capital
        self.capital = max(
            0, (1 - self.depreciation_rate) * self.capital + investment + gov_investment
        )
        self.capital = min(self.capital, self.max_capital)

        # Calculate utility
        reward = self._calculate_utility(consumption, labor_supply)

        # Prepare state and info
        state = self._get_state()
        info = {
            "investment": investment,
            "consumption": consumption,
            "utility": reward,
            "output": output,
        }

        done = False
        return state, reward, done, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.capital = 1.0
        self.tax_rate = 0.2
        self.gov_spending = 0.2
        self.money_supply = 1.0
        self.technology = 0.0
        return self._get_state(), {}

    def _calculate_output(self, labor: float) -> float:
        return np.exp(self.technology) * (
            self.capital ** self.capital_share_of_output
        ) * (labor ** (1 - self.capital_share_of_output))

    def _calculate_utility(self, consumption: float, labor: float) -> float:
        return self.utility_function(consumption, labor, **self.utility_params)

    def render(self):
        print("State:", self._get_state())

    def _get_state(self) -> Dict:
        return {
            "Capital": self.capital,
            "TaxRate": self.tax_rate,
            "GovSpending": self.gov_spending,
            "MoneySupply": self.money_supply,
        }

    def analytical_step(self) -> Tuple[Dict, float, bool, bool, Dict]:
        # Implement an analytical solution for testing purposes.
        return self.step(self.action_space.sample())

    @property
    def state_description(self):
        return {
            "Capital": "Capital stock",
            "TaxRate": "Tax rate on output",
            "GovSpending": "Government spending",
            "MoneySupply": "Total money supply",
        }

    @property
    def action_description(self):
        return {
            "investment_rate": "Fraction of output allocated to investment",
            "leisure": "Time allocated to leisure",
            "tax_rate_change": "Adjustment to tax rate",
            "gov_spending_change": "Adjustment to government spending",
            "money_supply_change": "Adjustment to money supply",
        }
