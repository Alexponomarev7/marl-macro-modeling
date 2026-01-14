import numpy as np
from typing import Union


def crra(c: float, sigma: float = 1.0) -> float:
    """
    CRRA utility function.
    
    U(c) = log(c)           if sigma = 1
    U(c) = (c^(1-σ) - 1)/(1-σ)  otherwise
    
    :param c: consumption, must be positive
    :param sigma: coefficient of relative risk aversion (CRRA parameter)
    :return: utility value
    """
    if c <= 0:
        return -np.inf
    if np.isclose(sigma, 1.0):
        return np.log(c)
    return c ** (1 - sigma) / (1 - sigma)


def cara(c: float, sigma: float = 1.0) -> float:
    """
    CARA utility function.
    
    U(c) = -exp(-σ*c) / η  or  1 - exp(-σ*c) (monotonic transform)
    
    :param c: consumption, must be non-negative
    :param sigma: coefficient of absolute risk aversion
    :return: utility value
    """
    if c < 0:
        return -np.inf
    return 1 - np.exp(-sigma * c)


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
