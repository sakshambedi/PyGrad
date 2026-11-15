"""Regression tests for two bugs found while comparing PyGrad against PyTorch.

These need no optional dependency, so they run in every environment.
"""

import pytest

from grad.tensor import Tensor
from grad.utils.misc import broadcast_shape


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ((2, 3, 1), (3, 2), [2, 3, 2]),
        ((3, 2), (2, 3, 1), [2, 3, 2]),
        ((4, 1, 3), (2, 1), [4, 2, 3]),
        ((2, 3), (3,), [2, 3]),
        ((3,), (2, 3), [2, 3]),
        ((), (2, 3), [2, 3]),
    ],
)
def test_broadcast_shape_with_different_ranks(left, right, expected):
    # Used to return e.g. [3, 2, 3, 2] for (2, 3, 1) and (3, 2).
    assert broadcast_shape(left, right) == expected


def test_binary_op_broadcasts_different_ranks():
    a = Tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])  # (2, 3, 1)
    b = Tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])  # (3, 2)
    out = a * b
    assert out.shape == (2, 3, 2)
    assert out.tolist() == [
        [[10.0, 20.0], [60.0, 80.0], [150.0, 180.0]],
        [[40.0, 80.0], [150.0, 200.0], [300.0, 360.0]],
    ]


def test_binary_op_respects_transposed_view():
    # Used to read the view's storage in its original order: [[1, 2], [3, 4], [5, 6]].
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).transpose(0, 1)
    zeros = Tensor.zeros((3, 2))
    expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert (t + zeros).tolist() == expected
    assert (zeros + t).tolist() == expected


def test_unary_op_respects_transposed_view():
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).transpose(0, 1)
    assert (-t).tolist() == [[-1.0, -4.0], [-2.0, -5.0], [-3.0, -6.0]]


def test_unary_op_on_expanded_view():
    # Used to raise IndexError: the result had 6 elements' shape but 2 of storage.
    e = Tensor([[1.0], [2.0]]).expand(2, 3)
    assert (-e).tolist() == [[-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0]]
