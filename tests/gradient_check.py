def finite_difference_grads(fn, a: float, b: float, eps: float = 1e-6):
    grad_a = (fn(a + eps, b) - fn(a - eps, b)) / (2 * eps)
    grad_b = (fn(a, b + eps) - fn(a, b - eps)) / (2 * eps)
    return grad_a, grad_b
