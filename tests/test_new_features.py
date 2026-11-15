"""Tests for rand, eye, mean, matmul, exp, log, backward, and reverse operators."""

import math

import pytest

from grad.dtype import dtypes
from grad.tensor import Tensor


class TestRand:
    def test_shape(self):
        t = Tensor.rand(3, 4)
        assert t.shape == (3, 4)

    def test_values_in_range(self):
        t = Tensor.rand(100)
        for i in range(100):
            assert 0.0 <= t[(i,)] < 1.0

    def test_dtype(self):
        t = Tensor.rand(2, 2, dtype=dtypes.float64)
        assert t.dtype == dtypes.float64

    def test_scalar(self):
        t = Tensor.rand()
        assert t.shape == ()


class TestEye:
    def test_square_identity(self):
        t = Tensor.eye(3)
        assert t.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                assert t[(i, j)] == pytest.approx(1.0 if i == j else 0.0)

    @pytest.mark.parametrize("n,m", [(2, 4), (4, 2), (1, 3)])
    def test_rectangular(self, n, m):
        t = Tensor.eye(n, m)
        assert t.shape == (n, m)
        for i in range(n):
            for j in range(m):
                assert t[(i, j)] == pytest.approx(1.0 if i == j else 0.0)

    def test_dtype(self):
        t = Tensor.eye(2, dtype=dtypes.float64)
        assert t.dtype == dtypes.float64

    def test_single(self):
        t = Tensor.eye(1)
        assert t.shape == (1, 1)
        assert t[(0, 0)] == pytest.approx(1.0)


# ── Mean ─────────────────────────────────────────────────────────────────────


class TestMean:
    def test_global(self, assert_tensor_close):
        t = Tensor([1, 2, 3, 4], dtype=dtypes.float32)
        assert t.mean().item() == pytest.approx(2.5)

    @pytest.mark.parametrize(
        "dim, expected",
        [
            (0, [2.0, 3.0]),
            (1, [1.5, 3.5]),
            (-1, [1.5, 3.5]),
        ],
    )
    def test_along_dim(self, dim, expected, assert_tensor_close):
        t = Tensor([[1, 2], [3, 4]], dtype=dtypes.float32)
        assert_tensor_close(t.mean(dim=dim), expected)

    def test_keepdims(self):
        t = Tensor([[1, 2], [3, 4]], dtype=dtypes.float32)
        m = t.mean(dim=0, keepdims=True)
        assert m.shape == (1, 2)

    def test_negative_dim(self, assert_tensor_close):
        t = Tensor([[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32)
        assert_tensor_close(t.mean(dim=-1), [2.0, 5.0])


# ── Matmul ───────────────────────────────────────────────────────────────────


class TestMatmul:
    def test_2d_2d(self, assert_tensor_close):
        a = Tensor([[1, 2], [3, 4]], dtype=dtypes.float32)
        b = Tensor([[5, 6], [7, 8]], dtype=dtypes.float32)
        assert_tensor_close(Tensor.matmul(a, b), [[19, 22], [43, 50]])

    def test_operator(self, assert_tensor_close):
        a = Tensor([[1, 0], [0, 1]], dtype=dtypes.float32)
        b = Tensor([[5, 6], [7, 8]], dtype=dtypes.float32)
        assert_tensor_close(a @ b, [[5, 6], [7, 8]])

    def test_1d_dot_product(self):
        a = Tensor([1, 2, 3], dtype=dtypes.float32)
        b = Tensor([4, 5, 6], dtype=dtypes.float32)
        assert Tensor.matmul(a, b).item() == pytest.approx(32.0)

    @pytest.mark.parametrize(
        "a_data, a_shape, b_data, b_shape, expected",
        [
            ([[1, 2], [3, 4]], (2, 2), [1, 1], (2,), [3.0, 7.0]),
            ([1, 2], (2,), [[1, 0], [0, 1]], (2, 2), [1.0, 2.0]),
        ],
        ids=["2d_x_1d", "1d_x_2d"],
    )
    def test_mixed_dims(self, a_data, a_shape, b_data, b_shape, expected, assert_tensor_close):
        a = Tensor(a_data, dtype=dtypes.float32)
        b = Tensor(b_data, dtype=dtypes.float32)
        assert_tensor_close(Tensor.matmul(a, b), expected)

    def test_shape_mismatch_raises(self):
        a = Tensor([[1, 2, 3]], dtype=dtypes.float32)
        b = Tensor([[1, 2]], dtype=dtypes.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            Tensor.matmul(a, b)

    def test_eye_is_identity(self, assert_tensor_close):
        a = Tensor([[1, 2], [3, 4]], dtype=dtypes.float32)
        assert_tensor_close(a @ Tensor.eye(2), [[1, 2], [3, 4]])


# ── Exp / Log ────────────────────────────────────────────────────────────────


class TestExpLog:
    @pytest.mark.parametrize(
        "inputs, expected_fn",
        [
            ([0, 1, 2], math.exp),
            ([[0, 1], [2, 3]], math.exp),
        ],
        ids=["1d", "2d"],
    )
    def test_exp(self, inputs, expected_fn, assert_tensor_close):
        t = Tensor(inputs, dtype=dtypes.float32)
        expected = [expected_fn(v) for v in Tensor._flatten_gen(inputs)]
        assert_tensor_close(t.exp(), Tensor._nest(expected, list(t.shape)), atol=1e-4)

    def test_exp_scalar(self):
        assert Tensor(0.0, dtype=dtypes.float32).exp().item() == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "inputs, expected_fn",
        [
            ([1, math.e, math.e**2], math.log),
            ([[1, 2], [3, 4]], math.log),
        ],
        ids=["1d", "2d"],
    )
    def test_log(self, inputs, expected_fn, assert_tensor_close):
        t = Tensor(inputs, dtype=dtypes.float64)
        expected = [expected_fn(v) for v in Tensor._flatten_gen(inputs)]
        assert_tensor_close(t.log(), Tensor._nest(expected, list(t.shape)), atol=1e-6)

    def test_exp_log_roundtrip(self, assert_tensor_close):
        t = Tensor([1.0, 2.0, 3.0], dtype=dtypes.float64)
        assert_tensor_close(t.exp().log(), [1.0, 2.0, 3.0], atol=1e-6)


# ── Reverse operators ───────────────────────────────────────────────────────
class TestReverseOps:
    @pytest.mark.parametrize(
        "scalar, op, data, expected",
        [
            (5, "__radd__", [1, 2, 3], [6, 7, 8]),
            (10, "__rsub__", [1, 2, 3], [9, 8, 7]),
            (3, "__rmul__", [1, 2, 3], [3, 6, 9]),
            (8, "__rtruediv__", [1, 2, 4], [8, 4, 2]),
            (2, "__rpow__", [1, 2, 3], [2, 4, 8]),
        ],
        ids=["radd", "rsub", "rmul", "rtruediv", "rpow"],
    )
    def test_reverse_op(self, scalar, op, data, expected, assert_tensor_close):
        t = Tensor(data, dtype=dtypes.float32)
        result = getattr(t, op)(scalar)
        assert_tensor_close(result, expected)


# ── Backward / Autograd ─────────────────────────────────────────────────────
class TestBackward:
    @pytest.mark.parametrize(
        "a_data, b_data, op, grad_a, grad_b",
        [
            ([2.0, 3.0], [4.0, 5.0], "add", [1.0, 1.0], [1.0, 1.0]),
            ([5.0, 6.0], [1.0, 2.0], "sub", [1.0, 1.0], [-1.0, -1.0]),
            ([2.0, 3.0], [4.0, 5.0], "mul", [4.0, 5.0], [2.0, 3.0]),
            ([6.0, 8.0], [3.0, 2.0], "div", [1 / 3, 1 / 2], [-6 / 9, -8 / 4]),
        ],
        ids=["add", "sub", "mul", "div"],
    )
    def test_binary_op_backward(self, a_data, b_data, op, grad_a, grad_b, assert_tensor_close):
        a = Tensor(a_data, dtype=dtypes.float32, requires_grad=True)
        b = Tensor(b_data, dtype=dtypes.float32, requires_grad=True)

        ops = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}
        ops[op].sum().backward()

        assert_tensor_close(a.grad, grad_a)
        assert_tensor_close(b.grad, grad_b)

    def test_neg_backward(self, assert_tensor_close):
        a = Tensor([1.0, 2.0], dtype=dtypes.float32, requires_grad=True)
        (-a).sum().backward()
        assert_tensor_close(a.grad, [-1.0, -1.0])

    def test_pow_backward(self, assert_tensor_close):
        a = Tensor([2.0, 3.0], dtype=dtypes.float64, requires_grad=True)
        (a**2).sum().backward()
        assert_tensor_close(a.grad, [4.0, 6.0])

    def test_exp_backward(self, assert_tensor_close):
        a = Tensor([1.0, 2.0], dtype=dtypes.float32, requires_grad=True)
        a.exp().sum().backward()
        assert_tensor_close(a.grad, [math.exp(1.0), math.exp(2.0)], atol=1e-4)

    def test_log_backward(self, assert_tensor_close):
        a = Tensor([1.0, 2.0, 4.0], dtype=dtypes.float64, requires_grad=True)
        a.log().sum().backward()
        assert_tensor_close(a.grad, [1.0, 0.5, 0.25], atol=1e-6)

    def test_chain_mul_add(self):
        """Gradient through (a * b) + c."""
        a = Tensor([2.0], dtype=dtypes.float32, requires_grad=True)
        b = Tensor([3.0], dtype=dtypes.float32, requires_grad=True)
        c = Tensor([1.0], dtype=dtypes.float32, requires_grad=True)
        (a * b + c).sum().backward()
        assert a.grad[(0,)] == pytest.approx(3.0)  # d/da = b
        assert b.grad[(0,)] == pytest.approx(2.0)  # d/db = a
        assert c.grad[(0,)] == pytest.approx(1.0)  # d/dc = 1

    def test_no_grad_raises(self):
        a = Tensor([1.0], dtype=dtypes.float32, requires_grad=False)
        with pytest.raises(RuntimeError, match="doesn't require grad"):
            a.backward()

    def test_non_scalar_without_gradient_raises(self):
        a = Tensor([1.0, 2.0], dtype=dtypes.float32, requires_grad=True)
        b = Tensor([3.0, 4.0], dtype=dtypes.float32, requires_grad=True)
        with pytest.raises(RuntimeError, match="non-scalar"):
            (a + b).backward()
