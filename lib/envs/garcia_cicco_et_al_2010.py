import warnings; warnings.filterwarnings("ignore")

import numpy as np
import gymnasium as gym

from lib.envs.environment_base import ENV_TO_ID, AbstractEconomicEnv


class GarciaCiccoEnv(AbstractEconomicEnv):
    """
    Garcia-Cicco et al. (2010) model environment
    """

    def __init__(
            self,
            start_capital: float = 0.1,
            alpha: float = 0.32,
            beta: float = 0.98**4,
            delta: float = 1.03**4-1,
            gamma_a: float = 2.0,
            omega: float = 1.6,
            theta: float = 1.4 * 1.6,
            phi: float = 4.81,
            psi: float = 2.87,
            dbar: float = 0.0,
            gbar: float = 1.0,
            rho_a: float = 0.0,
            rho_g: float = 0.0,
            rho_nu: float = 0.0,
            rho_mu: float = 0.0,
            rho_s: float = 0.0,
            **kwargs,
    ) -> None:
        super().__init__()

        # Model parameters
        self.start_capital = start_capital
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma_a = gamma_a
        self.omega = omega
        self.theta = 1.4 * omega if theta is None else theta
        self.phi = phi
        self.psi = psi
        self.dbar = dbar
        self.gbar = gbar

        # Shock persistence
        self.rho_a = rho_a
        self.rho_g = rho_g
        self.rho_nu = rho_nu
        self.rho_mu = rho_mu
        self.rho_s = rho_s

        # State variables
        self.capital = start_capital
        self.debt = 0.0
        self.productivity = 0.0
        self.hours = 0.0
        self.output = 0.0
        self.consumption = 0.0
        self.investment = 0.0
        self.trade_balance = 1.0
        self.growth = self.gbar
        self.interest_rate = 0.0
        self.country_premium = 1.0
        self.preference = 1.0
        self.investment_shock = 0.0

        # Spaces
        self.observation_space = gym.spaces.Dict({
            "Capital": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "Debt": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "LoggedProductivity": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "HoursWorked": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "Output": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "Consumption": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "Investment": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "TradeBalance": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "TechGrowthRate": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "InterestRate": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "CountryPremiumShock": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "PreferenceShock": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "Spending": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        })

        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -np.inf]),
            high=np.array([np.inf, np.inf]),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset environment"""
        super().reset(seed=seed)
        self.current_step = 0

        # Reset state variables
        self.capital = self.start_capital
        self.debt = 0.0
        self.productivity = 0.0
        self.hours = 0.0
        self.output = 0.0
        self.consumption = 0.0
        self.investment = 0.0
        self.trade_balance = np.exp(self.output - self.consumption - self.investment)
        self.growth = self.gbar
        self.interest_rate = 1/self.beta * self.gbar**self.gamma_a
        self.country_premium = 1.0
        self.preference = 1.0
        self.investment_shock = 0.0

        return self._get_state(), {}

    def step(self, Consumption: float, HoursWorked: float, Investment: float) -> tuple[dict, float, bool, bool, dict]:
        """Advance model"""
        self.consumption = Consumption
        self.hours = HoursWorked
        self.investment = Investment

        # Step 1: output from production function
        self.output = np.exp(self.productivity) * self.capital**self.alpha * (self.growth * self.hours)**(1 - self.alpha)
        self.capital =  ((1 - self.delta) * self.capital + self.investment) / self.growth

        self.trade_balance = np.exp(self.output - self.consumption - self.investment)
        self.debt = (self.debt * self.interest_rate / self.growth) + (self.consumption + self.investment - self.output)

        # Step 2: update interest rate
        self.interest_rate = 1/self.beta * self.gbar**self.gamma_a + self.psi * (np.exp(self.debt - self.dbar) - 1)

        reward = self.preference * (self.consumption - self.theta/self.omega * self.hours**self.omega)**(1-self.gamma_a)
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
            "Capital": self.capital,
            "Debt": self.debt,
            "LoggedProductivity": self.productivity,
            "HoursWorked": self.hours,
            "Output": self.output,
            "Consumption": self.consumption,
            "Investment": self.investment,
            "TradeBalance": self.trade_balance,
            "TechGrowthRate": self.growth,
            "InterestRate": self.interest_rate,
            "CountryPremiumShock": self.country_premium,
            "PreferenceShock": self.preference,
        }

    def analytical_step(self) -> tuple[dict, float, bool, bool, dict]:
        raise NotImplementedError("Analytical step is not implemented for Garcia-Cicco et al. (2010) model")

    def render(self):
        return super().render()

    def close(self):
        return super().close()

    @property
    def task_id(self) -> int:
        return ENV_TO_ID["GarciaCicco"]

    @property
    def params(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "delta": self.delta,
            "gamma_a": self.gamma_a,
            "omega": self.omega,
            "theta": self.theta,
            "phi": self.phi,
            "psi": self.psi,
            "dbar": self.dbar,
            "gbar": self.gbar,
            "rho_a": self.rho_a,
            "rho_g": self.rho_g,
            "rho_nu": self.rho_nu,
            "rho_mu": self.rho_mu,
            "rho_s": self.rho_s,
        }

    @property
    def state_description(self) -> dict[str, str]:
        return {
            "Capital": "Capital stock",
            "Debt": "External debt",
            "LoggedProductivity": "Total factor productivity (in logs)",
            "HoursWorked": "Hours worked",
            "Output": "Output",
            "Consumption": "Consumption",
            "Investment": "Investment",
            "TradeBalance": "Trade balance",
            "TechGrowthRate": "Technology growth rate",
            "InterestRate": "Interest rate",
            "CountryPremiumShock": "Country premium shock",
            "PreferenceShock": "Preference shock"
        }

    @property
    def action_description(self) -> dict[str, str]:
        return {
            "Consumption": "Consumption choice",
            "HoursWorked": "Hours worked",
            "Investment": "Investment choice",
        }