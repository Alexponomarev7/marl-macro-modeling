import numpy as np
from typing import Union
import pandas as pd


def crra(c: float, theta: float = 0.99):
    return (c ** (1 - theta) - 1) / (1 - theta)


def log_utility(C: Union[float, np.array], L: Union[float, np.array], A: float = 1.0) -> Union[float, np.array]:
    """
    Calculate the log utility for consumption and labor in RBC model.
    U(C,L) = ln(C) + A*ln(1-L)

    :param C: consumption, must be positive
    :param L: labor supply, must be between 0 and 1
    :param A: weight on leisure in utility function, defaults to 1.0
    :return: utility value
    """
    # if np.any(L >= 1) or np.any(L <= 0):
    #     raise ValueError("Labor supply must be between 0 and 1")
    # if np.any(C <= 0):
    #     raise ValueError("Consumption must be positive")

    return np.log(C) + A * np.log(1 - L)


def ces_utility(
        C: Union[float, np.array],
        L: Union[float, np.array],
        sigma: float = 2.0,
        eta: float = 1.5,
        A: float = 1.0
) -> Union[float, np.array]:
    """
    Calculate the CES utility for consumption and labor in RBC model.
    U(C,L) = [C^(1-σ)]/(1-σ) + A*[(1-L)^(1-η)]/(1-η)

    :param C: consumption, must be positive
    :param L: labor supply, must be between 0 and 1
    :param sigma: coefficient of relative risk aversion, defaults to 2.0
    :param eta: inverse Frisch elasticity of labor supply, defaults to 1.5
    :param A: weight on leisure in utility function, defaults to 1.0
    :return: utility value
    """
    # if np.any(L >= 1) or np.any(L <= 0):
    #     raise ValueError("Labor supply must be between 0 and 1")
    # if np.any(C <= 0):
    #     raise ValueError("Consumption must be positive")

    consumption_utility = (C ** (1 - sigma)) / (1 - sigma)
    leisure_utility = A * ((1 - L) ** (1 - eta)) / (1 - eta)
    return consumption_utility + leisure_utility



def calculate_utility(df: pd.DataFrame, beta: float = 0.9, sigma: float = 1.0, phi: float=0.8) -> pd.Series:
    """
    Рассчитывает ожидаемую дисконтированную полезность для каждого шага симуляции.

    :param df: DataFrame с данными симуляции.
    :param beta: Коэффициент дисконтирования.
    :param sigma: Параметр неприятия риска.
    :param phi: Параметр эластичности предложения труда.
    :return: Series с ожидаемой полезностью для каждого шага.
    """
    # Извлекаем данные

    if "Utility" in df.columns:
        return df["Utility"]

    C = df['Consumption'].values  # Потребление
    N = df['Labor'].values  # Отработанные часы
    T = len(C)  # Количество периодов

    # Вектор для хранения полезности
    utility = np.zeros(T)

    # Рассчитываем полезность для каждого шага
    for t in range(T):
        # Сумма полезностей для всех будущих периодов
        discounted_sum = 0.0
        for s in range(T - t):
            # Полезность в периоде t+s
            if sigma == 1:
                u_t_s = np.log(C[t + s]) - (N[t + s] ** (1 + phi)) / (1 + phi)
            else:
                u_t_s = (C[t + s] ** (1 - sigma)) / (1 - sigma) - (N[t + s] ** (1 + phi)) / (1 + phi)

            # Дисконтированная полезность
            discounted_sum += (beta ** s) * u_t_s

        # Сохраняем результат
        utility[t] = discounted_sum

    return pd.Series(utility, index=df.index)

def calculate_macro_utility(
    df: pd.DataFrame,
    utility_type: str = 'ces',
    params: dict = None
) -> pd.Series:
    """
    Calculate utility from macroeconomic simulation data with different specifications.

    Args:
        df: DataFrame with simulation data containing at least 'Consumption' and 'Labor' columns
        utility_type: Type of utility function to use ('ces', 'log', 'crra')
        params: Dictionary of parameters for the utility function

    Returns:
        pd.Series with utility values for each period
    """
    if params is None:
        params = {}

    # Default parameters
    default_params = {
        'beta': 0.99,  # discount factor
        'sigma': 2.0,  # coefficient of relative risk aversion
        'eta': 1.5,    # inverse Frisch elasticity
        'A': 1.0,      # weight on leisure
        'theta': 0.99  # CRRA parameter
    }

    # Update with provided parameters
    params = {**default_params, **params}

    # Extract data
    C = df['Consumption'].values
    L = df['Labor'].values
    T = len(C)

    # Initialize utility array
    utility = np.zeros(T)

    # Calculate period utility based on type
    if utility_type == 'ces':
        period_utility = ces_utility(C, L,
                                   sigma=params['sigma'],
                                   eta=params['eta'],
                                   A=params['A'])
    elif utility_type == 'log':
        period_utility = log_utility(C, L, A=params['A'])
    elif utility_type == 'crra':
        period_utility = crra(C, theta=params['theta'])
    else:
        raise ValueError(f"Unknown utility type: {utility_type}")

    # Calculate discounted utility
    for t in range(T):
        # Sum of discounted utilities for all future periods
        discounted_sum = 0.0
        for s in range(T - t):
            discounted_sum += (params['beta'] ** s) * period_utility[t + s]
        utility[t] = discounted_sum

    return pd.Series(utility, index=df.index)

def calculate_macro_reward(
    df: pd.DataFrame,
    utility_type: str = 'ces',
    params: dict = None,
    normalize: bool = False
) -> pd.Series:
    """
    Calculate reward from macroeconomic simulation data using utility function.
    Optionally normalizes the reward to have zero mean and unit variance.

    Args:
        df: DataFrame with simulation data
        utility_type: Type of utility function to use
        params: Parameters for the utility function
        normalize: Whether to normalize the reward

    Returns:
        pd.Series with reward values for each period
    """
    # Calculate utility
    utility = calculate_macro_utility(df, utility_type, params)

    # Calculate reward as change in utility
    reward = utility.diff()

    # Handle first period
    reward.iloc[0] = 0

    # Normalize if requested
    if normalize:
        reward = (reward - reward.mean()) / reward.std()

    return reward