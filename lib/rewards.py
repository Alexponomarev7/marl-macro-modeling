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
    U(c) = -exp(-sigma * c) / sigma

    Or equivalently (monotonic transformation):
    U(c) = 1 - exp(-sigma * c)

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
    U = log(c1_t) + beta * log(c2_{t+1})

    Note: In OLG models, beta represents discounting between youth and old age,
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
    parameters: dict[str, float],
    consumption_column: str = 'Consumption',
    labor_column: str = 'Labor',
    preference_shock_column: str | None = 'PreferenceShock',
    gamma_column: str | None = None,
    theta_column: str | None = None,
    omega_column: str | None = None,
    gamma_default: float = 2.0,
    theta_default: float = 2.24,
    omega_default: float = 1.6,
) -> pd.Series:
    """
    GHH (Greenwood-Hercowitz-Huffman) utility reward.
    Used in Garcia-Cicco, Pancrazi, Uribe (2010).
    
    U(C, H) = nu * [C - theta / omega * H^omega]^(1-gamma) / (1-gamma)
    
    Key feature: No wealth effect on labor supply (C and H are not separable).
    
    Args:
        data: DataFrame with simulation data
        parameters: Model parameters from Dynare
        consumption_column: Column name for consumption
        labor_column: Column name for hours worked
        preference_shock_column: Column name for preference shock nu (or None)
        gamma_column: Parameter name for risk aversion gamma
        theta_column: Parameter name for labor utility weight theta
        omega_column: Parameter name for labor disutility curvature omega
        gamma_default: Default value for gamma
        theta_default: Default value for theta (typically 1.4 * omega)
        omega_default: Default value for omega
    
    Returns:
        pd.Series with utility values for each period
    """
    C = data[consumption_column]
    H = data[labor_column]
    
    if preference_shock_column and preference_shock_column in data.columns:
        nu = data[preference_shock_column]
    else:
        nu = 1.0
    
    gamma = (
        data[gamma_column]
        if gamma_column and gamma_column in data.columns
        else parameters.get(gamma_column, gamma_default)
        if gamma_column
        else parameters.get('gamma_c', gamma_default)
    )
    
    theta = (
        data[theta_column]
        if theta_column and theta_column in data.columns
        else parameters.get(theta_column, theta_default)
        if theta_column
        else parameters.get('theta', theta_default)
    )
    
    omega = (
        data[omega_column]
        if omega_column and omega_column in data.columns
        else parameters.get(omega_column, omega_default)
        if omega_column
        else parameters.get('omega', omega_default)
    )
    
    consumption_equiv = C - (theta / omega) * (H ** omega)
    consumption_equiv = np.maximum(consumption_equiv, 1e-10)
    
    if isinstance(gamma, (int, float)):
        if np.isclose(gamma, 1.0):
            utility = nu * np.log(consumption_equiv)
        else:
            utility = nu * (consumption_equiv ** (1 - gamma)) / (1 - gamma)
    else:
        utility = pd.Series(index=data.index, dtype=float)
        log_mask = np.isclose(gamma, 1.0, atol=1e-6)
        utility[log_mask] = nu * np.log(consumption_equiv[log_mask])
        utility[~log_mask] = (
            nu * (consumption_equiv[~log_mask] ** (1 - gamma[~log_mask])) 
            / (1 - gamma[~log_mask])
        )
    
    utility = utility.replace([np.inf, -np.inf], np.nan)
    utility = utility.fillna(-1e6)
    
    return utility


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
    sigma_default: float = 1.0,
    eta_default: float = 1.0,
    A_default: float = 1.0,
) -> pd.Series:
    """
    CES utility reward:
    U(C,L) = C^(1-sigma) / (1-sigma) + A * (1-L)^(1-η) / (1-η)
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


def government_welfare(
    data: pd.DataFrame,
    parameters: dict[str, float],
    consumption_column: str = 'Consumption',
    labor_column: str = 'Labor',
    output_column: str = 'Output',
    gov_spending_column: str = 'GovSpending',
    sigma_column: str | None = 'sigma',
    lambda_utility: float = 1.0,
    lambda_output_gap: float = 0.1,
    lambda_smoothing: float = 0.05,
) -> pd.Series:
    """
    Функция благосостояния правительства.

    W = lambda_u * U(C,L) - lambda_y * (Y/Y_ss - 1)^2 - lambda_g * (ΔG/G_ss)^2

    Args:
        data: DataFrame с данными симуляции
        parameters: Параметры модели из Dynare
        consumption_column: Название колонки потребления
        labor_column: Название колонки труда
        output_column: Название колонки выпуска
        gov_spending_column: Название колонки госрасходов
        sigma_column: Название параметра CRRA (или None для default=1)
        lambda_utility: Вес полезности домохозяйств
        lambda_output_gap: Вес стабилизации выпуска
        lambda_smoothing: Вес сглаживания госрасходов
    
    Returns:
        pd.Series с значениями welfare для каждого периода
    """
    C = data[consumption_column]
    L = data[labor_column]
    Y = data[output_column]
    G = data[gov_spending_column]
    
    if sigma_column and sigma_column in parameters:
        sigma = parameters[sigma_column]
    else:
        sigma = 1.0
    
    psi = parameters.get('psi', 1.0)
    y_ss = parameters.get('y_ss', Y.mean())
    g_ss = parameters.get('g_ss', G.mean())

    if np.isclose(sigma, 1.0):
        consumption_utility = np.log(C)
    else:
        consumption_utility = (C ** (1 - sigma)) / (1 - sigma)
    
    leisure = np.maximum(1 - L, 1e-10)
    leisure_utility = psi * np.log(leisure)
    household_utility = consumption_utility + leisure_utility

    output_gap = ((Y - y_ss) / y_ss) ** 2

    g_change = (G.diff().fillna(0) / g_ss) ** 2
    
    welfare = (
        lambda_utility * household_utility 
        - lambda_output_gap * output_gap 
        - lambda_smoothing * g_change
    )
    
    welfare = welfare.replace([np.inf, -np.inf], np.nan)
    welfare = welfare.fillna(-1e6)
    
    return welfare


def central_bank_loss(
    data: pd.DataFrame,
    parameters: dict[str, float],
    inflation_column: str = 'price_inflation',
    output_gap_column: str = 'output_gap',
    lambda_y: float = 0.5,
    inflation_target: float = 0.0,
) -> pd.Series:
    """
    Central bank quadratic loss function.
    
    L = π² + λ_y * y_gap²
    
    Returns -L as reward (minimizing loss = maximizing negative loss).
    
    Args:
        data: DataFrame with simulation data
        parameters: Model parameters
        inflation_column: Column name for inflation
        output_gap_column: Column name for output gap
        lambda_y: Weight on output gap stabilization
        inflation_target: Target inflation rate (usually 0 in linear models)
    
    Returns:
        pd.Series with reward (negative loss) for each period
    """
    pi = data[inflation_column]
    y_gap = data[output_gap_column]
    
    loss = (pi - inflation_target)**2 + lambda_y * y_gap**2
    reward = -loss
    
    reward = reward.replace([np.inf, -np.inf], np.nan)
    reward = reward.fillna(-1e6)
    
    return reward
