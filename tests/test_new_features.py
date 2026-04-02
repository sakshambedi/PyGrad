"""Tests for newly implemented features: rand, eye, mean, matmul, exp, log,
backward, and reverse operators."""

import math

import pytest

from grad.dtype import dtypes
from grad.tensor import Tensor


def _assert_tensor_close(tensor, expected_list, atol=1e-5):
    """Assert tensor values match expected flat/nested list within tolerance."""
    from grad.utils.misc import _nd_indices

    if isinstance(expected_list, (int, float)):
        assert tensor.item() == pytest.approx(expected_list, abs=atol)
        return
    flat_expected = list(Tensor._flatten_gen(expected_list))
    flat_actual = [tensor[idx] for idx in _nd_indices(tensor.shape)]
    assert len(flat_actual) == len(flat_expected)
    for a, e in zip(flat_actual, flat_expected):
        assert a == pytest.approx(e, abs=atol), f"got {a}, expected {e}"


# ── Factory methods ──────────────────────────────────────────────────────────


class TestRand:
    def test_shape(self):
        t = Tensor.rand(3, 4)
        assert t.shape == (3, 4)

    def test_values_in_range(self):
        t = Tensor.rand(100)
        for i in range(100):
            v = t[(i,)]
            assert 0.0 <= v < 1.0

    def test_dtype(self):
        t = Tensor.rand(2, 2, dtype=dtypes.float64)
        assert t.dtype == dtypes.float64

    def test_empty_shape_scalar(self):
        t = Tensor.rand()
        assert t.shape == ()


class TestEye:
    def test_square(self):
        t = Tensor.eye(3)
        assert t.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert t[(i, j)] == pytest.approx(expected)

    def test_rectangular(self):
        t = Tensor.eye(2, 4)
        assert t.shape == (2, 4)
        assert t[(0, 0)] == pytest.approx(1.0)
        assert t[(1, 1)] == pytest.approx(1.0)
        assert t[(0, 2)] == pytest.approx(0.0)
        assert t[(1, 3)] == pytest.approx(0.0)

    def test_dtype(self):
        t = Tensor.eye(2, dtype=dtypes.float64)
        assert t.dtype == dtypes.float64

    def test_single(self):
        t = Tensor.eye(1)
        assert t.shape == (1, 1)
        assert t[(0, 0)] == pytest.approx(1.0)


# ── Mean ─────────────────────────────────────────────────────────────────────


class TestMean:
    def test_global_mean(self):
        t = Tensor([1, 2, 3, 4], dtype=dtypes.float32)
        m = t.mean()
        assert m.item() == pytest.approx(2.5)

    def test_mean_dim0(self):
        t = Tensor([[1, 2], [3, 4]], dtype=dtypes.float32)
        m = t.mean(dim=0)
        _assert_tensor_close(m, [2.0, 3.0])

    def test_mean_dim1(self):
        t = Tensor([[1, 2], [3, 4]], dtype=dtypes.float32)
        m = t.mean(dim=1)
        _assert_tensor_close(m, [1.5, 3.5])

    def test_mean_keepdims(self):
        t = Tensor([[1, 2], [3, 4]], dtype=dtypes.float32)
        m = t.mean(dim=0, keepdims=True)
        assert m.shape == (1, 2)

    def test_mean_negative_dim(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32)
        m = t.mean(dim=-1)
        _assert_tensor_close(m, [2.0, 5.0])


# ── Matmul ───────────────────────────────────────────────────────────────────


class TestMatmul:
    def test_2d_matmul(self):
        a = Tensor([[1, 2], [3, 4]], dtype=dtypes.float32)
        b = Tensor([[5, 6], [7, 8]], dtype=dtypes.float32)
        c = Tensor.matmul(a, b)
        _assert_tensor_close(c, [[19, 22], [43, 50]])

    def test_matmul_operator(self):
        a = Tensor([[1, 0], [0, 1]], dtype=dtypes.float32)
        b = Tensor([[5, 6], [7, 8]], dtype=dtypes.float32)
        c = a @ b
        _assert_tensor_close(c, [[5, 6], [7, 8]])

    def test_1d_dot(self):
        a = Tensor([1, 2, 3], dtype=dtypes.float32)
        b = Tensor([4, 5, 6], dtype=dtypes.float32)
        c = Tensor.matmul(a, b)
        assert c.item() == pytest.approx(32.0)

    def test_2d_1d(self):
        a = Tensor([[1, 2], [3, 4]], dtype=dtypes.float32)
        b = Tensor([1, 1], dtype=dtypes.float32)
        c = Tensor.matmul(a, b)
        _assert_tensor_close(c, [3.0, 7.0])

    def test_1d_2d(self):
        a = Tensor([1, 2], dtype=dtypes.float32)
        b = Tensor([[1, 0], [0, 1]], dtype=dtypes.float32)
        c = Tensor.matmul(a, b)
        _assert_tensor_close(c, [1.0, 2.0])

    def test_shape_mismatch_raises(self):
        a = Tensor([[1, 2, 3]], dtype=dtypes.float32)
        b = Tensor([[1, 2]], dtype=dtypes.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            Tensor.matmul(a, b)

    def test_eye_matmul_identity(self):
        a = Tensor([[1, 2], [3, 4]], dtype=dtypes.float32)
        eye = Tensor.eye(2)
        c = a @ eye
        _assert_tensor_close(c, [[1, 2], [3, 4]])


# ── Exp / Log ────────────────────────────────────────────────────────────────


class TestExp:
    def test_exp_values(self):
        t = Tensor([0, 1, 2], dtype=dtypes.float32)
        r = t.exp()
        _assert_tensor_close(r, [math.exp(0), math.exp(1), math.exp(2)])

    def test_exp_2d(self):
        t = Tensor([[0, 1], [2, 3]], dtype=dtypes.float32)
        r = t.exp()
        _assert_tensor_close(r, [[math.exp(0), math.exp(1)], [math.exp(2), math.exp(3)]], atol=1e-4)

    def test_exp_scalar(self):
        t = Tensor(0.0, dtype=dtypes.float32)
        r = t.exp()
        assert r.item() == pytest.approx(1.0)


class TestLog:
    def test_log_values(self):
        t = Tensor([1, math.e, math.e**2], dtype=dtypes.float64)
        r = t.log()
        _assert_tensor_close(r, [0.0, 1.0, 2.0], atol=1e-6)

    def test_log_2d(self):
        t = Tensor([[1, 2], [3, 4]], dtype=dtypes.float64)
        r = t.log()
        _assert_tensor_close(r, [[math.log(1), math.log(2)], [math.log(3), math.log(4)]], atol=1e-6)


# ── Reverse operators ───────────────────────────────────────────────────────


class TestReverseOps:
    def test_radd(self):
        t = Tensor([1, 2, 3], dtype=dtypes.float32)
        r = 5 + t
        _assert_tensor_close(r, [6, 7, 8])

    def test_rsub(self):
        t = Tensor([1, 2, 3], dtype=dtypes.float32)
        r = 10 - t
        _assert_tensor_close(r, [9, 8, 7])

    def test_rmul(self):
        t = Tensor([1, 2, 3], dtype=dtypes.float32)
        r = 3 * t
        _assert_tensor_close(r, [3, 6, 9])

    def test_rtruediv(self):
        t = Tensor([1, 2, 4], dtype=dtypes.float32)
        r = 8 / t
        _assert_tensor_close(r, [8, 4, 2])

    def test_rpow(self):
        t = Tensor([1, 2, 3], dtype=dtypes.float32)
        r = 2 ** t
        _assert_tensor_close(r, [2, 4, 8])


# ── Backward / Autograd ─────────────────────────────────────────────────────


class TestBackward:
    def test_add_backward(self):
        a = Tensor([2.0, 3.0], dtype=dtypes.float32, requires_grad=True)
        b = Tensor([4.0, 5.0], dtype=dtypes.float32, requires_grad=True)
        c = a + b
        c_sum = c.sum()
        c_sum.backward()
        _assert_tensor_close(a.grad, [1.0, 1.0])
        _assert_tensor_close(b.grad, [1.0, 1.0])

    def test_mul_backward(self):
        a = Tensor([2.0, 3.0], dtype=dtypes.float32, requires_grad=True)
        b = Tensor([4.0, 5.0], dtype=dtypes.float32, requires_grad=True)
        c = a * b
        c_sum = c.sum()
        c_sum.backward()
        _assert_tensor_close(a.grad, [4.0, 5.0])
        _assert_tensor_close(b.grad, [2.0, 3.0])

    def test_sub_backward(self):
        a = Tensor([5.0, 6.0], dtype=dtypes.float32, requires_grad=True)
        b = Tensor([1.0, 2.0], dtype=dtypes.float32, requires_grad=True)
        c = a - b
        c_sum = c.sum()
        c_sum.backward()
        _assert_tensor_close(a.grad, [1.0, 1.0])
        _assert_tensor_close(b.grad, [-1.0, -1.0])

    def test_div_backward(self):
        a = Tensor([6.0, 8.0], dtype=dtypes.float32, requires_grad=True)
        b = Tensor([3.0, 2.0], dtype=dtypes.float32, requires_grad=True)
        c = a / b
        c_sum = c.sum()
        c_sum.backward()
        _assert_tensor_close(a.grad, [1 / 3, 1 / 2])
        _assert_tensor_close(b.grad, [-6 / 9, -8 / 4])

    def test_neg_backward(self):
        a = Tensor([1.0, 2.0], dtype=dtypes.float32, requires_grad=True)
        c = -a
        c_sum = c.sum()
        c_sum.backward()
        _assert_tensor_close(a.grad, [-1.0, -1.0])

    def test_chain_backward(self):
        """Test gradient through a chain: (a * b) + c."""
        a = Tensor([2.0], dtype=dtypes.float32, requires_grad=True)
        b = Tensor([3.0], dtype=dtypes.float32, requires_grad=True)
        c = Tensor([1.0], dtype=dtypes.float32, requires_grad=True)
        d = a * b + c
        d_sum = d.sum()
        d_sum.backward()
        assert a.grad[(0,)] == pytest.approx(3.0)
        assert b.grad[(0,)] == pytest.approx(2.0)
        assert c.grad[(0,)] == pytest.approx(1.0)

    def test_exp_backward(self):
        a = Tensor([1.0, 2.0], dtype=dtypes.float32, requires_grad=True)
        c = a.exp()
        c_sum = c.sum()
        c_sum.backward()
        _assert_tensor_close(a.grad, [math.exp(1.0), math.exp(2.0)], atol=1e-4)

    def test_log_backward(self):
        a = Tensor([1.0, 2.0, 4.0], dtype=dtypes.float64, requires_grad=True)
        c = a.log()
        c_sum = c.sum()
        c_sum.backward()
        _assert_tensor_close(a.grad, [1.0, 0.5, 0.25], atol=1e-6)

    def test_backward_no_grad_raises(self):
        a = Tensor([1.0], dtype=dtypes.float32, requires_grad=False)
        with pytest.raises(RuntimeError, match="doesn't require grad"):
            a.backward()

    def test_backward_non_scalar_no_gradient_raises(self):
        a = Tensor([1.0, 2.0], dtype=dtypes.float32, requires_grad=True)
        b = Tensor([3.0, 4.0], dtype=dtypes.float32, requires_grad=True)
        c = a + b
        with pytest.raises(RuntimeError, match="non-scalar"):
            c.backward()

    def test_pow_backward(self):
        a = Tensor([2.0, 3.0], dtype=dtypes.float64, requires_grad=True)
        c = a ** 2
        c_sum = c.sum()
        c_sum.backward()
        _assert_tensor_close(a.grad, [4.0, 6.0])
