import math

import pytest

from tests.gradient_check import finite_difference_grads

try:
    from grad.autograd.ops import Div, Mul, Neg, Pow
except ImportError:  # pragma: no cover - depends on compiled extension availability
    Div = Mul = Neg = Pow = None

pytestmark = pytest.mark.skipif(Div is None, reason="autograd ops extension is unavailable")


class DummyContext:
    def __init__(self, *saved):
        self.saved_tensor = saved


def test_mul_backward_matches_formula():
    ctx = DummyContext(2.0, 3.0)
    grad_a, grad_b = Mul.backward(ctx, 4.0)
    assert grad_a == pytest.approx(12.0)
    assert grad_b == pytest.approx(8.0)


def test_div_backward_matches_formula():
    ctx = DummyContext(6.0, 3.0)
    grad_a, grad_b = Div.backward(ctx, 2.0)
    assert grad_a == pytest.approx(2.0 / 3.0)
    assert grad_b == pytest.approx(-(6.0 * 2.0) / (3.0**2))


def test_div_backward_handles_zero_denominator():
    ctx = DummyContext(1.0, 0.0)
    grad_a, grad_b = Div.backward(ctx, 1.0)
    assert math.isinf(grad_a)
    assert math.isinf(grad_b)


def test_pow_backward_matches_finite_difference():
    a, b = 2.5, 1.5
    grad_output = 1.0
    ctx = DummyContext(a, b)
    grad_a, grad_b = Pow.backward(ctx, grad_output)
    num_grad_a, num_grad_b = finite_difference_grads(lambda x, y: x**y, a, b)
    assert grad_a == pytest.approx(num_grad_a, rel=1e-5, abs=1e-6)
    assert grad_b == pytest.approx(num_grad_b, rel=1e-5, abs=1e-6)


def test_neg_backward_matches_formula():
    (grad_a,) = Neg.backward(DummyContext(5.0), 3.0)
    assert grad_a == pytest.approx(-3.0)


def test_gradient_flows_through_mul_and_div_chain():
    a, b, c = 1.5, 2.0, 0.5
    mul_ctx = DummyContext(a, b)
    div_ctx = DummyContext(a * b, c)

    grad_u, grad_c = Div.backward(div_ctx, 1.0)
    grad_a, grad_b = Mul.backward(mul_ctx, grad_u)

    assert grad_a == pytest.approx(b / c)
    assert grad_b == pytest.approx(a / c)
    assert grad_c == pytest.approx(-(a * b) / (c**2))
