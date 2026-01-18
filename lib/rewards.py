import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.

    All reward functions must implement the compute method with the signature:
    - data: pd.DataFrame - The data containing state/action variables
    - parameters: dict[str, float] - Model parameters
    - **kwargs - Additional configuration parameters (e.g., target_column, sigma_column, etc.)

    Reward functions should accept **kwargs and ignore any parameters they don't use.
    """

    @abstractmethod
    def __call__(
        self,
        data: pd.DataFrame,
        parameters: dict[str, float],
        **kwargs
    ) -> pd.Series:
        """
        Compute reward for each row in the data.

        Args:
            data: DataFrame with state/action variables
            parameters: Model parameters dictionary
            **kwargs: Additional configuration (target_column, sigma_column, etc.)

        Returns:
            Series of reward values, one per row
        """
        pass


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
    target_indices: list[int] | None = None,
    **kwargs  # Accept any additional kwargs (e.g., weights) and ignore them
) -> pd.Series:
    """
    Stability reward: returns the target column value directly.

    Args:
        data: DataFrame with state/action variables
        parameters: Model parameters (unused)
        target_column: Column name to use as reward
        target_indices: Alternative to target_column - list of column indices to use as reward.
                       If provided and target_column is None, uses the first index.
        **kwargs: Additional parameters (ignored)
    """
    if target_column is not None:
        return data[target_column]
    elif target_indices is not None and len(target_indices) > 0:
        # Use the first target index to get the column name
        col_idx = target_indices[0]
        column_name = data.columns[col_idx]
        return data[column_name]
    else:
        raise ValueError("Either target_column or target_indices must be provided")


def log_reward(
    data: pd.DataFrame,
    parameters: dict[str, float],
    target_column: str = "Consumption",
    **kwargs  # Accept any additional kwargs and ignore them
) -> pd.Series:
    """
    Log utility: U(C) = ln(C)
    This is CRRA with sigma = 1.

    Args:
        data: DataFrame with state/action variables
        parameters: Model parameters (unused)
        target_column: Column name for consumption
        **kwargs: Additional parameters (ignored)
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
    **kwargs  # Accept any additional kwargs and ignore them
) -> pd.Series:
    """
    CRRA (Constant Relative Risk Aversion) utility reward:
    U(c) = c^(1-σ)/(1-σ) for σ ≠ 1, or U(c) = ln(c) for σ = 1

    Args:
        data: DataFrame with state/action variables
        parameters: Model parameters
        target_column: Column name for consumption
        sigma_column: Column name or parameter name for risk aversion coefficient
        sigma_default: Default value for sigma
        **kwargs: Additional parameters (ignored)
    """
    assert target_column is not None, "target_column must be provided"

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
    **kwargs  # Accept any additional kwargs and ignore them
) -> pd.Series:
    """
    CARA (Constant Absolute Risk Aversion) utility reward:
    U(c) = -exp(-σ * c) / σ

    Or equivalently (monotonic transformation):
    U(c) = 1 - exp(-σ * c)

    Args:
        data: DataFrame with consumption data
        parameters: Model parameters
        target_column: Column name for consumption
        sigma_column: Column name or parameter name for risk aversion coefficient
        sigma_default: Default value for sigma
        **kwargs: Additional parameters (ignored)
    """
    assert target_column is not None, "target_column must be provided"

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
    **kwargs  # Accept any additional kwargs and ignore them
) -> pd.Series:
    """
    OLG (Overlapping Generations) log utility reward:
    U = log(c1_t) + β * log(c2_{t+1})

    Note: In OLG models, β represents discounting between youth and old age,
    not between periods as in Ramsey model.

    Args:
        data: DataFrame with consumption data
        parameters: Model parameters including beta
        consumption_young_column: Column name for young consumption
        consumption_old_column: Column name for old consumption
        beta_column: Column name or parameter name for discount factor
        beta_default: Default value for beta
        **kwargs: Additional parameters (ignored)
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
    parameters: dict[str, float],
    **kwargs  # Accept any additional kwargs and ignore them
) -> pd.Series:
    """
    Garcia-Cicco et al. (2010) utility reward function.

    Args:
        data: DataFrame with Consumption, HoursWorked, PreferenceShock columns
        parameters: Model parameters (theta, omega, gamma_a)
        **kwargs: Additional parameters (ignored)
    """
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
    **kwargs  # Accept any additional kwargs and ignore them
) -> pd.Series:
    """
    Log utility reward:
    U(C,L) = ln(C) + A * ln(1-L)

    Args:
        data: DataFrame with state/action variables
        parameters: Model parameters
        consumption_column: Column name for consumption
        labor_column: Column name for labor
        A_column: Column name or parameter name for A coefficient
        A_default: Default value for A
        **kwargs: Additional parameters (ignored)
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
    **kwargs  # Accept any additional kwargs and ignore them
) -> pd.Series:
    """
    CES utility reward:
    U(C,L) = C^(1-σ)/(1-σ) + A * (1-L)^(1-η)/(1-η)

    Args:
        data: DataFrame with state/action variables
        parameters: Model parameters
        consumption_column: Column name for consumption
        labor_column: Column name for labor
        sigma_column: Column name or parameter name for sigma
        eta_column: Column name or parameter name for eta
        A_column: Column name or parameter name for A
        sigma_default: Default value for sigma
        eta_default: Default value for eta
        A_default: Default value for A
        **kwargs: Additional parameters (ignored)
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
