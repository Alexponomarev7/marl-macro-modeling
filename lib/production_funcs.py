def cobb_douglas(K, L, A: float = 1.0, alpha: float = 0.5, beta: float = 0.5):
    return A * K**alpha * L**beta
