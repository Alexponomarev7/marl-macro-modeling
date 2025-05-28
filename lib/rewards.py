import numpy as np
import pandas as pd
from typing import Optional


def l1_norm(state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
    """
    Вычисляет L1-норму разницы между текущим и следующим состоянием.
    """
    return -np.sum(np.abs(next_state - state))


def l2_norm(state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
    """
    Вычисляет L2-норму (евклидову норму) разницы между текущим и следующим состоянием.
    """
    return -np.sqrt(np.sum((next_state - state) ** 2))


def column_reward(state: pd.Series, target_column: str) -> float:
    return state[target_column]


def utility_reward(
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    utility_index: int,
) -> float:
    """
    Вычисляет награду на основе полезности (utility).

    Args:
        state: Текущее состояние.
        action: Действие.
        next_state: Следующее состояние.
        utility_index: Индекс полезности в массиве состояния.

    Returns:
        Значение полезности в следующем состоянии.
    """
    return next_state[utility_index][0]