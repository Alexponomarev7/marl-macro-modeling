def crra(c: float, theta: float = 0.99):
    return (c**(1-theta) - 1) / (1 - theta)
