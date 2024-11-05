import gymnasium as gym
from utility_funcs import crra
from production_funcs import cobb_douglas

class NCGEnv(gym.Env):
    def __init__(self, initial_capital: float = 1.0, deprecation: float = 0.9):
        self.capital = initial_capital
        self.deprecation = deprecation

    def step(self, action):
        assert action >= 0 and action <= 1, "action must be in [0, 1]"
        consumption = self.capital * action

        self.capital -= consumption
        self.capital = self.capital * self.deprecation + cobb_douglas(self.capital, 1)

        reward = crra(consumption)
        return self.capital, reward, False, False, {}
