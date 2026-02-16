import numpy as np
import pytest

from grad.autograd.ops import Add
from grad.tensor import Tensor


class TestAddition:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([1, 2, 3], [4, 5, 6]),
            ([1, 2, 3], [0.5, 1.0, 1.5]),
            ([1.5, 2.5, 3.5], [1, 2, 3]),
            ([[1, 2], [3, 4]], [[10, 20], [30, 40]]),
            ([[0, 1, 2], [3, 4, 5]], [[1, 1, 1], [1, 1, 1]]),
            ([-1, -2, -3], [1, 2, 3]),
            ([1.5, 2.5, 3.5], [0.5, 1.0, 1.5]),
        ],
    )
    def test_add_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a + t_b
        res = np.array(a) + np.array(b)
        np.testing.assert_array_equal(t_c.to_numpy(), res)

    def test_add_2d(self):
        t0 = Tensor([[1, 2, 3], [4, 5, 6]])
        t1 = Tensor([[10, 20, 30], [40, 50, 60]])
        out = t0 + t1
        expected = np.array([[11, 22, 33], [44, 55, 66]])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_broadcast_add(self):
        t0 = Tensor.ones((3, 1))
        t1 = Tensor.arange(3)
        out = t0 + t1
        expected = np.ones((3, 1)) + np.arange(3)
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_broadcast_add_mismatched(self):
        t0 = Tensor.ones((2, 3))
        t1 = Tensor.arange(3)
        out = t0 + t1
        expected = np.ones((2, 3)) + np.arange(3)
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_broadcast_add_rank_mismatch_regression(self):
        t0 = Tensor.ones((3, 1))
        t1 = Tensor.arange(3)
        out = t0 + t1
        expected = np.ones((3, 1)) + np.arange(3)
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_add_shape_mismatch(self):
        t0 = Tensor([1, 2, 3])
        t1 = Tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="not broadcast-compatible"):
            _ = t0 + t1

    # # @requires_broadcasting_support
    def test_add_scalar(self):
        t = Tensor([1, 2, 3])
        result = t + 2
        assert result.shape == (3,)
        np.testing.assert_array_equal(result.to_numpy(), np.array([3, 4, 5]))

    def test_add_requires_grad(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6])
        result = t1 + t2
        assert result.requires_grad is True

    def test_add_backward_broadcast_reduces_to_input_shapes(self):
        ctx = Add()
        a = Tensor.ones((3, 1))
        b = Tensor.arange(3)

        out = Add.forward(ctx, a, b)
        grad_output = Tensor.ones(out.shape)
        grad_a, grad_b = Add.backward(ctx, grad_output)

        assert grad_a.shape == (3, 1)
        assert grad_b.shape == (3,)
        np.testing.assert_array_equal(grad_a.to_numpy(), np.full((3, 1), 3.0))
        np.testing.assert_array_equal(grad_b.to_numpy(), np.full((3,), 3.0))


class TestSubtraction:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([4, 5, 6], [1, 2, 3]),
            ([[10, 20], [30, 40]], [[1, 2], [3, 4]]),
            ([[1, 2, 3], [4, 5, 6]], [[1, 1, 1], [1, 1, 1]]),
            ([5, -6], [-1, -2]),
            ([2.5, 3.5], [0.5, 1.0]),
        ],
    )
    def test_sub_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a - t_b
        res = np.array(a) - np.array(b)
        np.testing.assert_array_equal(t_c.to_numpy(), res)

    def test_sub_2d(self):
        t0 = Tensor([[10, 20, 30], [40, 50, 60]])
        t1 = Tensor([[1, 2, 3], [4, 5, 6]])
        out = t0 - t1
        expected = np.array([[9, 18, 27], [36, 45, 54]])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_sub_shape_mismatch(self):
        t0 = Tensor([1, 2, 3])
        t1 = Tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="not broadcast-compatible"):
            _ = t0 - t1

    # # @requires_broadcasting_support
    def test_sub_scalar(self):
        t = Tensor([4, 5, 6])
        result = t - 2
        assert result.shape == (3,)
        np.testing.assert_array_equal(result.to_numpy(), np.array([2, 3, 4]))

    def test_sub_requires_grad(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6])
        result = t1 - t2
        assert result.requires_grad is True


class TestMultiplication:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([1, 2, 3], [4, 5, 6]),
            ([[1, 2], [3, 4]], [[10, 20], [30, 40]]),
            ([[0, 1, 2], [3, 4, 5]], [[1, 1, 1], [1, 1, 1]]),
            ([-1, -2], [5, -6]),
            ([1.5, 2.5], [0.5, 1.0]),
            ([2, 3], [0.5, 1.5]),
        ],
    )
    def test_mul_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a * t_b
        res = np.array(a) * np.array(b)
        np.testing.assert_array_equal(t_c.to_numpy(), res)

    def test_mul_2d(self):
        t0 = Tensor([[1, 2, 3], [4, 5, 6]])
        t1 = Tensor([[10, 20, 30], [40, 50, 60]])
        out = t0 * t1
        expected = np.array([[10, 40, 90], [160, 250, 360]])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_mul_with_zeros(self):
        t0 = Tensor([0, 1, 2])
        t1 = Tensor([5, 0, 3])
        out = t0 * t1
        expected = np.array([0, 0, 6])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_mul_shape_mismatch(self):
        t0 = Tensor([1, 2, 3])
        t1 = Tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="not broadcast-compatible"):
            _ = t0 * t1

    # # @requires_broadcasting_support
    def test_mul_scalar(self):
        t = Tensor([1, 2, 3])
        result = t * 2
        assert result.shape == (3,)
        np.testing.assert_array_equal(result.to_numpy(), np.array([2, 4, 6]))

    def test_mul_requires_grad(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6])
        result = t1 * t2
        assert result.requires_grad is True


class TestDivision:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([4, 6, 8], [2, 2, 2]),
            ([[10, 20], [30, 40]], [[2, 4], [5, 8]]),
            ([[6, 8, 10], [12, 15, 18]], [[2, 2, 2], [3, 3, 3]]),
            ([1.5, 2.0], [0.5, 1.0]),
        ],
    )
    def test_div_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a / t_b
        res = np.array(a) / np.array(b)
        np.testing.assert_array_equal(t_c.to_numpy(), res)

    def test_div_2d(self):
        t0 = Tensor([[10, 20, 30], [40, 50, 60]])
        t1 = Tensor([[2, 4, 5], [8, 10, 12]])
        out = t0 / t1
        expected = np.array([[5.0, 5.0, 6.0], [5.0, 5.0, 5.0]])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_div_by_zero(self):
        t0 = Tensor([1, 2, 3])
        t1 = Tensor([1, 0, 3])
        out = t0 / t1
        # According to ops.py, division by zero returns inf
        expected = np.array([1.0, float("inf"), 1.0])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_div_shape_mismatch(self):
        t0 = Tensor([1, 2, 3])
        t1 = Tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="not broadcast-compatible"):
            _ = t0 / t1

    # # @requires_broadcasting_support
    def test_div_scalar(self):
        t = Tensor([4, 10, 18])
        result = t / 2
        assert result.shape == (3,)
        np.testing.assert_array_equal(result.to_numpy(), np.array([2.0, 5.0, 9.0]))

    def test_div_requires_grad(self):
        t1 = Tensor([4, 10, 18], requires_grad=True)
        t2 = Tensor([2, 5, 6])
        result = t1 / t2
        assert result.requires_grad is True


class TestPower:
    @pytest.mark.parametrize(
        "a,b",
        [
            ([2, 3, 4], [2, 2, 2]),
            ([[2, 3], [4, 5]], [[2, 3], [1, 2]]),
            ([1, 2, 3], [0, 1, 2]),
            ([2.0, 3.0], [0.5, 2.0]),
        ],
    )
    def test_pow_matches_numpy(self, a, b):
        t_a = Tensor(a)
        t_b = Tensor(b)
        t_c = t_a**t_b
        res = np.array(a) ** np.array(b)
        # Use almost_equal for floating point comparisons
        if any(isinstance(x, float) for x in a) or any(isinstance(x, float) for x in b):
            np.testing.assert_array_almost_equal(t_c.to_numpy(), res, decimal=6)
        else:
            np.testing.assert_array_equal(t_c.to_numpy(), res)

    def test_pow_2d(self):
        t0 = Tensor([[2, 3, 4], [5, 6, 7]])
        t1 = Tensor([[2, 2, 2], [1, 3, 2]])
        out = t0**t1
        expected = np.array([[4, 9, 16], [5, 216, 49]])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_pow_special_cases(self):
        # Test with base 0 and positive exponent
        t0 = Tensor([0, 1, 2])
        t1 = Tensor([2, 0, 3])
        out = t0**t1
        expected = np.array([0, 1, 8])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_pow_negative_base(self):
        t0 = Tensor([-2, -3])
        t1 = Tensor([2, 3])
        out = t0**t1
        expected = np.array([4, -27])
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_pow_negative_exponent(self):
        pytest.skip(
            "Temporarily skipped: scalar exponent broadcasting in cpp binary_op is pending."
        )
        t = Tensor([2, 4, 8])
        result = t**-1
        expected = np.array([0.5, 0.25, 0.125])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=6)

    def test_pow_float_exponent(self):
        pytest.skip(
            "Temporarily skipped: scalar exponent broadcasting in cpp binary_op is pending."
        )
        t = Tensor([4, 9, 16])
        result = t**0.5
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=6)

    def test_pow_requires_grad(self):
        t1 = Tensor([2.0, 3.0], requires_grad=True)
        t2 = Tensor([3, 2])
        result = t1**t2
        assert result.requires_grad is True

    # @requires_broadcasting_support
    def test_pow_shape_mismatch(self):
        t0 = Tensor([1, 2, 3])
        t1 = Tensor([[1, 2], [3, 4]])
        with pytest.raises(
            ValueError,
            match=r"Shapes \(3,\) and \(2, 2\) are not broadcast-compatible\.",
        ):
            _ = t0**t1

    # @requires_broadcasting_support
    def test_pow_scalar(self):
        t = Tensor([2, 3, 4])
        result = t**2
        assert result.shape == (3,)
        np.testing.assert_array_equal(result.to_numpy(), np.array([4, 9, 16]))


class TestNegation:
    @pytest.mark.parametrize(
        "a",
        [
            [1, 2, 3],
            [[1, 2], [3, 4]],
            [0, -1, 2],
            [-1.5, 2.5, -3.0],
            [[[-1, 2], [3, -4]], [[5, -6], [-7, 8]]],
        ],
    )
    def test_neg_matches_numpy(self, a):
        t_a = Tensor(a)
        t_neg = -t_a
        expected = -np.array(a)
        np.testing.assert_array_equal(t_neg.to_numpy(), expected)

    def test_neg_1d(self):
        t = Tensor([1, -2, 3, -4])
        result = -t
        expected = np.array([-1, 2, -3, 4])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_neg_2d(self):
        t = Tensor([[1, -2, 3], [-4, 5, -6]])
        result = -t
        expected = np.array([[-1, 2, -3], [4, -5, 6]])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_neg_3d(self):
        t = Tensor([[[1, -2], [3, -4]], [[-5, 6], [-7, 8]]])
        result = -t
        expected = np.array([[[-1, 2], [-3, 4]], [[5, -6], [7, -8]]])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_neg_zeros(self):
        t = Tensor([0, 0.0, -0])
        result = -t
        expected = np.array([0, -0.0, 0])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_neg_float_precision(self):
        t = Tensor([1.23456789, -9.87654321])
        result = -t
        expected = np.array([-1.23456789, 9.87654321])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_double_negation(self):
        t = Tensor([1, -2, 3])
        result = -(-t)
        expected = np.array([1, -2, 3])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_neg_large_values(self):
        t = Tensor([1e6, -1e6, 1e-6, -1e-6])
        result = -t
        expected = np.array([-1e6, 1e6, -1e-6, 1e-6])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=10)

    def test_neg_requires_grad(self):
        t = Tensor([1, -2, 3], requires_grad=True)
        result = -t
        assert result.requires_grad is True
