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
    return -np.linalg.norm(next_state - state)


def column_reward(state: pd.Series, target_column: str) -> float:
    return state[target_column]


def stability_reward(
    data: pd.DataFrame,
    parameters: dict[str, float],
    target_column: str | None = None,
) -> pd.Series:
    assert target_column is not None
    return data[target_column]


def log_reward(
    data: pd.DataFrame,
    parameters: dict[str, float],
    target_column: str = "Consumption",
) -> pd.Series:
    """
    Log utility: U(C) = ln(C)
    This is CRRA with sigma = 1.
    """
    consumption = data[target_column]

    utility = np.log(consumption)
    utility = utility.replace([np.inf, -np.inf], np.nan)
    utility = utility.fillna(-1e6)

    return utility


def crra_reward(
    data: pd.DataFrame,
    parameters: dict[str, float],
    target_column: str | None = None,
    sigma_column: str | None = None,
    sigma_default: float = 1.0,
) -> pd.Series:
    assert target_column is not None

    consumption = data[target_column]

    if sigma_column is not None and sigma_column in data.columns:
        sigma = data[sigma_column]
    elif sigma_column is not None and sigma_column in parameters:
        sigma = parameters[sigma_column]
    else:
        sigma = sigma_default

    if isinstance(sigma, (int, float)):
        if np.isclose(sigma, 1):
            utility = np.log(consumption)
        else:
            utility = consumption ** (1 - sigma) / (1 - sigma)
    else:
        utility = pd.Series(index=data.index, dtype=float)
        log_mask = np.isclose(sigma, 1.0, atol=1e-6)
        utility[log_mask] = np.log(consumption[log_mask])
        utility[~log_mask] = consumption[~log_mask] ** (1 - sigma[~log_mask])  / (1 - sigma[~log_mask])

    utility = utility.replace([np.inf, -np.inf], np.nan)
    utility = utility.fillna(-1e6)

    return utility


def cara_reward(
    data: pd.DataFrame,
    parameters: dict[str, float],
    target_column: str | None = None,
    sigma_column: str | None = None,
    sigma_default: float = 1.0,
) -> pd.Series:
    """
    CARA (Constant Absolute Risk Aversion) utility reward:
    U(c) = -exp(-σ * c) / σ

    Or equivalently (monotonic transformation):
    U(c) = 1 - exp(-σ * c)

    :param data: DataFrame with consumption data
    :param parameters: model parameters
    :param target_column: column name for consumption
    :param sigma_column: column name or parameter name for risk aversion coefficient
    :param sigma_default: default value for sigma
    :return: utility series
    """
    assert target_column is not None

    consumption = data[target_column]

    if sigma_column is not None and sigma_column in data.columns:
        sigma = data[sigma_column]
    elif sigma_column is not None and sigma_column in parameters:
        sigma = parameters[sigma_column]
    else:
        sigma = sigma_default

    utility = 1 - np.exp(-sigma * consumption)
    utility = utility.replace([np.inf, -np.inf], np.nan)
    utility = utility.fillna(-1e6)

    return utility


def olg_log_utility_reward(
    data: pd.DataFrame,
    parameters: dict[str, float],
    consumption_young_column: str = 'ConsYoung',
    consumption_old_column: str = 'ConsOld',
    beta_column: str | None = None,
    beta_default: float = 0.4,
) -> pd.Series:
    """
    OLG (Overlapping Generations) log utility reward:
    U = log(c1_t) + β * log(c2_{t+1})

    Note: In OLG models, β represents discounting between youth and old age,
    not between periods as in Ramsey model.

    :param data: DataFrame with consumption data
    :param parameters: model parameters including beta
    :param consumption_young_column: column name for young consumption
    :param consumption_old_column: column name for old consumption
    :param beta_column: column name or parameter name for discount factor
    :param beta_default: default value for beta
    :return: utility series
    """

    c1 = data[consumption_young_column]
    c2 = data[consumption_old_column]

    if beta_column is not None and beta_column in data.columns:
        beta = data[beta_column]
    elif beta_column is not None and beta_column in parameters:
        beta = parameters[beta_column]
    else:
        beta = beta_default
    
    utility = np.log(c1) + beta * np.log(c2)
    utility = utility.replace([np.inf, -np.inf], np.nan)
    utility = utility.fillna(-1e6)

    return utility


def GarciaCicco(
    data: pd.DataFrame,
    parameters: dict[str, float]
) -> pd.Series:
    theta = parameters["theta"]
    omega = parameters["omega"]
    gamma = parameters["gamma_a"]

    C = data["Consumption"].values
    H = data["HoursWorked"].values
    nu = data["PreferenceShock"].values

    consumption_equiv = C - (theta / omega) * (H ** omega)
    consumption_equiv = np.maximum(consumption_equiv, 1e-8)
    utility = nu * (consumption_equiv ** (1 - gamma))

    utility_series = pd.Series(utility, index=data.index, name='utility')

    return utility_series


def log_utility_reward(
    data: pd.DataFrame,
    parameters: dict[str, float],
    consumption_column: str = 'Consumption',
    labor_column: str = 'Population',
    A_column: str | None = None,
    A_default: float = 1.0,
) -> pd.Series:
    """
    Log utility reward:
    U(C,L) = ln(C) + A * ln(1-L)
    """
    C = data[consumption_column]
    L = data[labor_column]

    if A_column is not None and A_column in data.columns:
        A = data[A_column]
    elif A_column is not None and A_column in parameters:
        A = parameters[A_column]
    else:
        A = A_default

    utility = np.log(C) + A * np.log(1 - L)
    utility = utility.replace([np.inf, -np.inf], np.nan)
    utility = utility.fillna(-1e6)

    return utility


def ces_utility_reward(
    data: pd.DataFrame,
    parameters: dict[str, float],
    consumption_column: str = 'Consumption',
    labor_column: str = 'Labor',
    sigma_column: str | None = None,
    eta_column: str | None = None,
    A_column: str | None = None,
    sigma_default: float = 2.0,
    eta_default: float = 1.0,
    A_default: float = 1.0,
) -> pd.Series:
    """
    CES utility reward:
    U(C,L) = C^(1-σ)/(1-σ) + A * (1-L)^(1-η)/(1-η)
    """
    C = data[consumption_column]
    L = data[labor_column]

    sigma = (
        data[sigma_column]
        if sigma_column and sigma_column in data.columns
        else parameters.get(sigma_column, sigma_default)
        if sigma_column
        else sigma_default
    )

    eta = (
        data[eta_column]
        if eta_column and eta_column in data.columns
        else parameters.get(eta_column, eta_default)
        if eta_column
        else eta_default
    )

    A = (
        data[A_column]
        if A_column and A_column in data.columns
        else parameters.get(A_column, A_default)
        if A_column
        else A_default
    )

    if isinstance(sigma, (int, float)):
        if np.isclose(sigma, 1.0):
            consumption_utility = np.log(C)
        else:
            consumption_utility = C ** (1 - sigma) / (1 - sigma)
    else:
        consumption_utility = pd.Series(index=data.index, dtype=float)
        log_mask = np.isclose(sigma, 1.0, atol=1e-6)
        consumption_utility[log_mask] = np.log(C[log_mask])
        consumption_utility[~log_mask] = (
            C[~log_mask] ** (1 - sigma[~log_mask]) / (1 - sigma[~log_mask])
        )

    leisure = np.maximum(1 - L, 1e-10)
    
    if isinstance(eta, (int, float)):
        if np.isclose(eta, 1.0):
            leisure_utility = A * np.log(leisure)
        else:
            leisure_utility = A * leisure ** (1 - eta) / (1 - eta)
    else:
        leisure_utility = pd.Series(index=data.index, dtype=float)
        log_mask = np.isclose(eta, 1.0, atol=1e-6)
        leisure_utility[log_mask] = A * np.log(leisure[log_mask])
        leisure_utility[~log_mask] = (
            A * leisure[~log_mask] ** (1 - eta[~log_mask]) / (1 - eta[~log_mask])
        )

    utility = consumption_utility + leisure_utility
    utility = utility.replace([np.inf, -np.inf], np.nan)
    utility = utility.fillna(-1e6)

    return utility
