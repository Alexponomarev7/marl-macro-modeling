import gymnasium as gym
import numpy as np
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Optional,
    Dict,
    Tuple,
    Union,
)


class AbstractEconomicEnv(gym.Env, ABC):
    """
    Abstract base class for an economic environment.
    This class defines the necessary methods that any economic environment must implement.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to initial state.

        :param seed: Random seed for reproducibility
        :param options: Additional options for reset (unused)
        :return: Initial state observation and empty info dictionary
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
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
    def analytical_step(self) -> Tuple[Dict, float, bool, bool, Dict]:
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
    def params(self) -> Dict[str, Union[float, str, dict]]:
        """
        Get the current parameters of the economic environment.

        :return: Dictionary containing the current parameters of the environment
        """
        pass

    @property
    @abstractmethod
    def state_description(self):
        """
        Provide descriptions of state variables.

        :return: Dictionary mapping state variable names to their descriptions
        """
        pass

    @property
    @abstractmethod
    def action_description(self):
        """
        Provide descriptions of action variables.

        :return: Dictionary mapping action variable names to their descriptions
        """
        pass
