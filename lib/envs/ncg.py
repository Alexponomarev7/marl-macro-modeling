from typing import Any
import numpy as np
from lib.utility_funcs import crra
from lib.production_funcs import cobb_douglas
from lib.envs.environment_base import AbstractEconomicEnv

def bisection_solve(func, low, high, tol=1e-8, max_iter=200):
    """
    Simple bisection root finder.
    Assumes func(low) * func(high) < 0 (i.e., there's a sign change).
    """
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        val_mid = func(mid)
        # If close enough to zero, or interval is tiny, stop
        if abs(val_mid) < tol or (high - low) < tol:
            return mid
        # Narrow down the bracket
        if func(low) * val_mid > 0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)

class NCGEnv(AbstractEconomicEnv):
    """
    NCGEnv is a simple environment that models the life of an agent with a Cobb-Douglas production function and a CRRA utility function.
    The agent has a capital stock that deprecates at a fixed rate and can consume a fraction of its capital each period.
    The agent's utility is given by the CRRA function.
    """
    def __init__(self, initial_capital: float = 1.0, deprecation: float = 0.9):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.deprecation = deprecation
        self.current_step = 0

    def step(self, action) -> tuple[float, float, bool, dict]:
        assert action >= 0 and action <= 1, "action must be in [0, 1]"
        consumption = self.capital * action

        self.capital -= consumption
        self.capital = self.capital * self.deprecation + cobb_douglas(self.capital, 1)

        reward = crra(consumption)
        return self.capital, reward, False, False, {}

    def analytical_step(self) -> tuple[float, float, bool, bool, dict]:
        """
        Attempt an analytical (or semi-analytical) one-step solution from
        the current state. Here, we find the consumption that keeps capital
        at a 'steady state' next period: K_{t+1} = K_t.

        1) Solve δ x + f(x) = K_t for x (leftover capital).
        2) Implied consumption is C = K_t - x.
        3) Next capital is then the same as current capital (by construction).
        4) Utility is CRRA(C).

        Returns:
            next_capital (float): the next state for capital
            reward (float): the utility from consumption
            done (bool): end of episode?
            truncated (bool): environment cut short?
            info (dict): extra details
        """
        K_t = self.capital

        # Edge case: if K_t is extremely small, avoid negative or zero consumption
        if K_t < 1e-12:
            # No consumption, no capital to solve for
            c_star = 0.0
            reward = crra(c_star)
            return K_t, reward, False, False, {"note": "capital near zero"}

        # We want to solve δx + cobb_douglas(x,1) - K_t = 0
        def steady_state_eq(x):
            return self.deprecation * x + cobb_douglas(x, 1) - K_t

        # We bracket the solution between 0 and K_t (leftover cannot exceed total capital)
        # But to be safe, you might guess upper bound as K_t or slightly more
        x_low, x_high = 0.0, max(K_t, 1.0)

        # Check signs for bracket
        f_low = steady_state_eq(x_low)      # = -K_t, should be negative
        f_high = steady_state_eq(x_high)    # could be positive or negative, depending on K_t
        # If f_high is still negative, then it means we have an edge solution at x_high
        # For robust code, handle that scenario. We'll do a naive fix here:
        if f_low * f_high > 0:
            # Expand the bracket until sign changes or we give up
            while f_high < 0 and x_high < 1e6 * K_t:
                x_high *= 2
                f_high = steady_state_eq(x_high)
            # If that fails, we accept x_high as best guess

        x_star = bisection_solve(steady_state_eq, x_low, x_high)

        c_star = K_t - x_star  # consumption
        reward = crra(c_star)

        # The "analytical step" by definition sets next capital = K_t
        next_capital = K_t  # no change (steady state)
        done = False
        truncated = False

        self.current_step += 1
        self.capital = next_capital

        info = {
            "x_star": x_star,
            "consumption": c_star,
            "consumption_fraction": c_star / K_t if K_t > 0 else 0.0,
        }
        info["action"] = [info["consumption_fraction"]]

        return self._get_state(), reward, done, truncated, info

    def render(self):
        """
        Display current state of the environment.
        Prints all state variables with 4 decimal precision.
        """
        state = self._get_state()
        print("\nCurrent State:")
        for key, value in state.items():
            print(f"{key}: {value[0]:.4f}")

    def close(self):
        """Clean up resources"""
        # included for compatibility with the Gymnasium API
        pass

    @property
    def params(self) -> dict[str, Any]:
        """
        Provide parameters of the environment.

        :return: Dictionary containing environment parameters
        """
        return {
            "initial_capital": self.capital,
            "deprecation": self.deprecation
        }

    @property
    def state_description(self):
        """
        Provide descriptions of state variables.

        :return: Dictionary mapping state variable names to their descriptions
        """
        return {
            "capital": "Current capital stock"
        }

    @property
    def action_description(self):
        """
        Provide descriptions of action variables.

        :return: Dictionary mapping action variable names to their descriptions
        """
        return {
            "consumption_fraction": "Fraction of capital to consume"
        }
    
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """
        Reset the environment to initial state.

        :param seed: Random seed for reproducibility
        :param options: Additional options for reset (unused)
        :return: Initial state observation and empty info dictionary
        """
        super().reset(seed=seed)
        self.capital = self.initial_capital
        self.current_step = 0
        return self._get_state(), {}


    def _get_state(self) -> dict:
        """
        Construct the current state observation dictionary.

        :return: Dictionary containing current values of all state variables
        """
        return {
            "Capital": np.array([self.capital], dtype=np.float32),
        }

