import pytest

from grad.tensor import Tensor
from grad.utils.misc import _nd_indices


@pytest.fixture
def assert_tensor_close():
    """Fixture that returns a function to compare tensor values against a nested list."""

    def _check(tensor, expected, atol=1e-5):
        if isinstance(expected, (int, float)):
            assert tensor.item() == pytest.approx(expected, abs=atol)
            return
        flat_expected = list(Tensor._flatten_gen(expected))
        flat_actual = [tensor[idx] for idx in _nd_indices(tensor.shape)]
        assert len(flat_actual) == len(flat_expected), (
            f"size mismatch: tensor has {len(flat_actual)} elements, expected {len(flat_expected)}"
        )
        for i, (a, e) in enumerate(zip(flat_actual, flat_expected)):
            assert a == pytest.approx(e, abs=atol), (
                f"element {i}: got {a}, expected {e}"
            )

    return _check
