import numpy as np


def l1_norm(state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
    return -np.sum(np.abs(next_state - state))