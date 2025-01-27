import numpy as np
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


def stability_reward(
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    target_indices: list[int],
    weights: Optional[list[float]] = None,
) -> float:
    """
    Универсальная функция для расчета награды за стабильность целевых показателей.

    Args:
        state: Текущее состояние.
        action: Действие.
        next_state: Следующее состояние.
        target_indices: Индексы целевых показателей в массиве состояния.
        weights: Веса для каждого целевого показателя (по умолчанию равны 1).

    Returns:
        Награда, рассчитанная как взвешенная сумма изменений целевых показателей.
    """
    if weights is None:
        weights = [1.0] * len(target_indices)
    reward = 0.0
    for idx, weight in zip(target_indices, weights):
        reward -= weight * np.abs(next_state[idx] - state[idx])
    return reward


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