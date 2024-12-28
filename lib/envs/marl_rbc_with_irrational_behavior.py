import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional
from sympy import symbols, Eq, solve, log, diff


class MARLMacroeconomicEnv(gym.Env):
    """
    Agents:
        - Consumers: Maximize utility but can act irrationally with a probability.
        - Firms: Maximize profits, affected by consumer behavior and market conditions.
        - Government: Adjusts taxes to balance public revenue and welfare.
    State Space:
        - Aggregate capital (K)
        - Aggregate labor supply (L)
        - Technology level (A, in logs)
        - Aggregate output (Y)
    Action Space:
        - Consumers: Choose leisure (1 - labor).
        - Firms: Choose prices and wages.
        - Government: Set income and corporate tax rates.
    """

    def __init__(self, num_consumers=10, num_firms=5, irrational_prob=0.1, discount_rate=0.99,
                 depreciation_rate=0.025, capital_share=0.36, tech_persistence=0.95,
                 tech_variance=0.007, max_capital=10.0):
        super().__init__()

        # Environment parameters
        self.num_consumers = num_consumers
        self.num_firms = num_firms
        self.irrational_prob = irrational_prob  # Probability of irrational behavior
        self.discount_rate = discount_rate
        self.depreciation_rate = depreciation_rate
        self.capital_share = capital_share
        self.tech_persistence = tech_persistence
        self.tech_variance = tech_variance
        self.max_capital = max_capital

        # Initializing state variables
        self.capital = 1.0  # Initial capital stock
        self.technology = 0.0  # Initial technology level (log)
        self.labor = 0.5  # Initial labor supply
        self.output = self._calculate_output()

        # State space
        self.observation_space = gym.spaces.Dict({
            "Capital": gym.spaces.Box(low=0.0, high=max_capital, shape=(1,), dtype=np.float32),
            "Labor": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "Technology": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "Output": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        })

        # Action spaces
        self.action_space = gym.spaces.Dict({
            "Consumers": gym.spaces.Box(low=0, high=1, shape=(num_consumers, 1), dtype=np.float32),  # Leisure choices
            "Firms": gym.spaces.Box(low=0, high=1, shape=(num_firms, 2), dtype=np.float32),  # Prices, wages
            "Government": gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),  # Income, corporate taxes
        })

    def _calculate_output(self) -> float:
        """
        Cobb-Douglas production function: Y = A * K^α * L^(1-α)
        """
        return np.exp(self.technology) * (self.capital ** self.capital_share) * \
            (self.labor ** (1 - self.capital_share))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)

        # Reset state variables
        self.capital = 1.0
        self.technology = 0.0
        self.labor = 0.5
        self.output = self._calculate_output()

        return self._get_state(), {}

    def _get_state(self) -> Dict:
        """
        Return the current state of the environment.
        """
        return {
            "Capital": np.array([self.capital], dtype=np.float32),
            "Labor": np.array([self.labor], dtype=np.float32),
            "Technology": np.array([self.technology], dtype=np.float32),
            "Output": np.array([self.output], dtype=np.float32),
        }

    def _apply_irrational_behavior(self, actions: np.ndarray) -> np.ndarray:
        """
        Introduce randomness to simulate irrational behavior.

        :param actions: Array of agent actions.
        :return: Modified actions with some agents acting irrationally.
        """
        random_mask = np.random.rand(*actions.shape) < self.irrational_prob
        irrational_noise = np.random.uniform(-0.1, 0.1, size=actions.shape)
        return np.clip(actions + random_mask * irrational_noise, 0, 1)

    def step(self, actions: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute a step in the environment.

        :param actions: Dictionary containing actions for all agents.
        :return: Tuple with state, reward, done, truncated, and additional info.
        """
        # Unpack actions
        consumer_actions = actions["Consumers"]
        firm_actions = actions["Firms"]
        gov_action = actions["Government"]

        # Apply irrational behavior
        consumer_actions = self._apply_irrational_behavior(consumer_actions)
        firm_actions = self._apply_irrational_behavior(firm_actions)

        # Update labor supply based on consumer actions (average leisure)
        avg_leisure = np.mean(consumer_actions)
        self.labor = 1 - avg_leisure

        # Update technology (AR(1) process)
        tech_shock = np.random.normal(0, self.tech_variance)
        self.technology = self.tech_persistence * self.technology + tech_shock

        # Calculate output
        self.output = self._calculate_output()

        # Government policies
        income_tax, corp_tax = gov_action

        # Firm behavior: calculate wages and profits
        prices = firm_actions[:, 0]
        wages = firm_actions[:, 1]
        total_wages = np.sum(wages) * self.labor
        profits = (prices * self.output - total_wages) * (1 - corp_tax)

        # Consumer income
        consumer_income = total_wages * (1 - income_tax)

        # Update capital stock
        investment = profits / self.num_firms
        self.capital = (1 - self.depreciation_rate) * self.capital + investment
        self.capital = np.clip(self.capital, 0, self.max_capital)

        # Calculate rewards
        consumer_rewards = consumer_income - avg_leisure * total_wages
        firm_rewards = profits
        government_reward = income_tax * total_wages + corp_tax * profits

        rewards = {
            "Consumers": consumer_rewards,
            "Firms": np.sum(firm_rewards),
            "Government": government_reward,
        }

        # Episode termination status
        done = False  # Infinite-horizon setup

        # Additional info
        info = {
            "Investment": investment,
            "Consumption": self.output - investment,
            "Utility": rewards,
            "Output": self.output,
        }

        return self._get_state(), rewards, done, False, info

    def render(self, mode="human"):
        """
        Render the current state of the environment.
        """
        state = self._get_state()
        print("\n--- Current State ---")
        for key, value in state.items():
            print(f"{key}: {value[0]:.4f}")

    def close(self):
        """Close the environment."""
        pass

    def action_description(self):
        """
        Provide descriptions of action variables.

        :return: Dictionary mapping action variable names to their descriptions.
        """
        return {
            "Consumers": "Array of leisure choices for each consumer (1 - labor supply).",
            "Firms": "Array of [price, wage] pairs for each firm.",
            "Government": "Array of [income tax rate, corporate tax rate]."
        }

    def analytical_solution(self):
        """
        Compute the analytical equilibrium for one step.

        This function calculates optimal policies for agents based on closed-form
        solutions derived from the Cobb-Douglas production function and utility maximization.

        :return: Tuple of next state, reward, done, truncated, and info.
        """

        # Define symbols
        K, L, C, I, W, P, T_inc, T_corp = symbols('K L C I W P T_inc T_corp')

        # Cobb-Douglas production function
        A = np.exp(self.technology)
        alpha = self.capital_share
        Y = A * (K ** alpha) * (L ** (1 - alpha))

        # Budget constraints
        wage_income = W * L
        profits = P * Y - W * L
        consumer_income = wage_income * (1 - T_inc)
        gov_revenue = T_inc * wage_income + T_corp * profits

        # Utility function: log utility for simplicity
        U = log(C) - 0.5 * (1 - L) ** 2  # Utility of consumption and leisure

        # Equilibrium constraints
        eq_consumption = Eq(C + I, Y)  # Consumption + investment = output
        eq_capital = Eq(K, (1 - self.depreciation_rate) * K + I)  # Capital accumulation
        eq_labor_supply = Eq(diff(U, L), W * (1 - T_inc))  # Labor-leisure trade-off
        eq_wages_prices = Eq(W, P * (1 - alpha) * (K / L) ** alpha)  # Marginal productivity of labor

        # Solve for optimal variables
        solution = solve([eq_consumption, eq_capital, eq_labor_supply, eq_wages_prices],
                         (C, I, W, P))

        # Extract solutions
        optimal_C = float(solution[C])
        optimal_I = float(solution[I])
        optimal_W = float(solution[W])
        optimal_P = float(solution[P])

        # Update state variables
        self.capital = (1 - self.depreciation_rate) * self.capital + optimal_I
        self.capital = np.clip(self.capital, 0, self.max_capital)
        self.labor = 1 - optimal_W  # Approximation: labor is a function of wages
        self.technology = self.tech_persistence * self.technology + np.random.normal(0, self.tech_variance)
        self.output = self._calculate_output()

        # Calculate rewards
        consumer_reward = log(optimal_C) - 0.5 * (1 - self.labor) ** 2
        firm_reward = optimal_P * self.output - optimal_W * self.labor
        government_reward = gov_revenue

        rewards = {
            "Consumers": consumer_reward,
            "Firms": firm_reward,
            "Government": government_reward
        }

        # Done and info
        done = False  # Infinite horizon
        info = {
            "Optimal Consumption": optimal_C,
            "Optimal Investment": optimal_I,
            "Optimal Wages": optimal_W,
            "Optimal Prices": optimal_P,
            "Output": self.output
        }

        return self._get_state(), rewards, done, False, info
