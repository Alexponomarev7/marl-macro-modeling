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


def stability_reward(
    data: pd.DataFrame,
    target_column: str | None = None,
) -> pd.Series:
    assert target_column is not None
    return data[target_column]

def crra_reward(
    data: pd.DataFrame,
    target_column: str | None = None,
) -> pd.Series:
    assert target_column is not None
    return pd.Series(np.log(data[target_column]))

def GarciaCicco(
    data: pd.DataFrame,
    parameters: dict[str, float]
) -> pd.Series:
    theta = parameters["theta"]
    omega = parameters["omega"]
    gamma = parameters["gamma"]
    return data["PreferenceShock"] * (data["Consumption"] - theta / omega * data["HoursWorked"]**omega)**(1-gamma)
