import numpy as np
from typing import Union
from numpy.typing import ArrayLike


def cobb_douglas(
    K: Union[float, ArrayLike],
    L: Union[float, ArrayLike] = 1.0,
    A: float = 1.0,
    alpha: float = 0.33,
) -> Union[float, ArrayLike]:
    return A * (K ** alpha) * (L ** (1 - alpha))
