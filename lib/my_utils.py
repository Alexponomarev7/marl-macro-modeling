import numpy as np


def clipped_exponential(scale: float, low: float | None, high: float | None) -> float:
    """
    Sample from an exponential distribution with scale `scale`, clipped to the range [`low`, `high`].
    """
    return float(np.clip(np.random.exponential(scale), low, high))


def convert_numpy_to_python(obj):
    """Convert numpy types to native Python types"""
    import numpy as np
    if isinstance(obj, (np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    return obj
