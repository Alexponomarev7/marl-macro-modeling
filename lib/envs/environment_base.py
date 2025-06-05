import gymnasium as gym
import numpy as np
from abc import (
    ABC,
    abstractmethod,
)

ENV_TO_ID = {
    "Ramsey": 0,
    "RBC_baseline": 1,
}

class AbstractEconomicEnv(gym.Env, ABC):
    """
    Abstract base class for an economic environment.
    This class defines the necessary methods that any economic environment must implement.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__()

    @property
    @abstractmethod
    def task_id(self) -> int:
        """
        Get the current task ID.
        """
        pass

    @abstractmethod
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """
        Reset the environment to initial state.

        :param seed: Random seed for reproducibility
        :param options: Additional options for reset (unused)
        :return: Initial state observation and empty info dictionary
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one time step of the economic model.

        :param action: Array of action variables
        :return: Tuple containing:
                - state: Dictionary of current state variables
                - reward: Utility value for current period
                - terminated: Whether episode has ended
                - truncated: Whether episode was truncated
                - info: Dictionary with additional information
        """
        pass

    @abstractmethod
    def render(self):
        """
        Display current state of the environment.
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up resources"""
        pass

    @abstractmethod
    def analytical_step(self) -> tuple[dict, float, bool, bool, dict]:
        """
        Calculate the analytical solution for the economic model for one time step.

        :return: Tuple containing:
                - state: Dictionary of current state variables
                - reward: Utility value for current period
                - terminated: Whether episode has ended
                - truncated: Whether episode was truncated
                - info: Dictionary with additional information
        """
        pass

    @property
    @abstractmethod
    def params(self) -> dict[str, float | str | dict]:
        """
        Get the current parameters of the economic environment.

        :return: Dictionary containing the current parameters of the environment
        """
        pass

    @property
    @abstractmethod
    def state_description(self) -> dict[str, str]:
        """
        Provide descriptions of state variables.

        :return: Dictionary mapping state variable names to their descriptions
        """
        pass

    @property
    @abstractmethod
    def action_description(self) -> dict[str, str]:
        """
        Provide descriptions of action variables.

        :return: Dictionary mapping action variable names to their descriptions
        """
        pass
