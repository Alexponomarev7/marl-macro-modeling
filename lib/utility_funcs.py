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
    if np.any(L >= 1) or np.any(L <= 0):
        raise ValueError("Labor supply must be between 0 and 1")
    if np.any(C <= 0):
        raise ValueError("Consumption must be positive")

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
    if np.any(L >= 1) or np.any(L <= 0):
        raise ValueError("Labor supply must be between 0 and 1")
    if np.any(C <= 0):
        raise ValueError("Consumption must be positive")

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
    C = df['Consumption'].values  # Потребление
    N = df['Hours Worked'].values  # Отработанные часы
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