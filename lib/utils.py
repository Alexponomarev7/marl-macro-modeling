import numpy as np

def clipped_exponential(scale: float, low: float | None, high: float | None) -> float:
    """
    Sample from an exponential distribution with scale `scale`, clipped to the range [`low`, `high`].
    """
    return np.clip(np.random.exponential(scale), low, high)
