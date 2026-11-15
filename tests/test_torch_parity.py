"""Compare PyGrad against PyTorch on the same inputs.

Every test builds the same computation twice, once with ``grad.tensor.Tensor`` and
once with ``torch.Tensor``, from identical NumPy inputs, and asserts that forward
values and gradients agree in shape and in value within the tolerance below.

PyTorch is an optional dependency. Without it this whole module is skipped:

    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    uv run pytest tests/test_torch_parity.py -v

What is covered (forward value and gradient unless stated otherwise):

- element-wise ``+ - * / **`` on equal shapes and with broadcasting
- reflected operators with a Python scalar (``2 + t``, ``2 - t``, ``2 * t``,
  ``2 / t``, ``2 ** t``) and a Python scalar exponent (``t ** 3``)
- unary ``-t``, ``exp``, ``log``
- ``sum`` and ``mean``: global and along a dimension, with and without keepdims
- graphs that reuse a node, and a three-layer sigmoid network with an MSE loss
- ``matmul`` / ``@``: forward values only
- ``+ * / -t exp sum mean`` applied to ``transpose`` / ``permute`` / ``reshape`` /
  ``expand`` views: forward values only

Known gaps, recorded as strict ``xfail`` tests so they turn into failures (and
must be moved up) as soon as the feature is implemented:

- ``matmul`` has no backward: its result carries no ``grad_fn``.
- ``view`` / ``reshape`` / ``transpose`` / ``permute`` / ``expand`` do not
  propagate gradients: a view carries no ``grad_fn``.

Because ``matmul`` cannot be differentiated, the network test writes each linear
layer as a broadcast multiply followed by ``sum(1)``, which is the same
arithmetic and does go through PyGrad's autograd.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is not installed; parity tests skipped")

from grad.dtype import dtypes  # noqa: E402
from grad.tensor import Tensor  # noqa: E402

# Tolerances for |pygrad - torch| <= ATOL + RTOL * |torch|, per dtype.
TOLERANCE = {
    "float32": {"rtol": 1e-5, "atol": 1e-6},
    "float64": {"rtol": 1e-12, "atol": 1e-12},
}
DTYPES = {
    "float32": (dtypes.float32, torch.float32, np.float32),
    "float64": (dtypes.float64, torch.float64, np.float64),
}

pytestmark = pytest.mark.parametrize("dtype", ["float32", "float64"])


def _rng() -> np.random.Generator:
    return np.random.default_rng(1234)


def _pair(arr: np.ndarray, dtype: str, requires_grad: bool = True):
    """Build a PyGrad tensor and a torch tensor holding exactly the same values."""
    g_dtype, t_dtype, np_dtype = DTYPES[dtype]
    arr = np.asarray(arr, dtype=np_dtype)
    g = Tensor(arr.tolist(), dtype=g_dtype, requires_grad=requires_grad)
    t = torch.tensor(arr, dtype=t_dtype, requires_grad=requires_grad)
    return g, t


def _assert_same(g: Tensor, t: torch.Tensor, dtype: str, what: str) -> None:
    got = g.to_numpy()
    want = t.detach().numpy()
    assert got.shape == want.shape, f"{what}: shape {got.shape} != torch {want.shape}"
    np.testing.assert_allclose(got, want, err_msg=what, **TOLERANCE[dtype])


def _check(fn, inputs: list[np.ndarray], dtype: str) -> None:
    """Run ``fn`` on both libraries; compare the output and every input gradient.

    The output is back-propagated with a non-uniform upstream gradient so that a
    wrong Jacobian cannot hide behind a constant one.
    """
    pairs = [_pair(a, dtype) for a in inputs]
    g_out = fn(*[p[0] for p in pairs])
    t_out = fn(*[p[1] for p in pairs])
    _assert_same(g_out, t_out, dtype, "forward")

    upstream = _rng().uniform(0.5, 1.5, size=tuple(t_out.shape))
    g_up, t_up = _pair(upstream, dtype, requires_grad=False)
    g_out.backward(g_up)
    t_out.backward(t_up)

    for i, (g_in, t_in) in enumerate(pairs):
        assert g_in.grad is not None, f"input {i}: PyGrad produced no gradient"
        _assert_same(g_in.grad, t_in.grad, dtype, f"gradient of input {i}")


def _positive(*shape: int) -> np.ndarray:
    return _rng().uniform(0.5, 2.0, size=shape)


def _signed(*shape: int) -> np.ndarray:
    # Second generator state so the two operands of a binary op differ.
    rng = np.random.default_rng(4321)
    values = rng.uniform(0.5, 2.0, size=shape)
    return values * rng.choice([-1.0, 1.0], size=shape)


BINARY_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "pow": lambda a, b: a**b,
}

# (left shape, right shape); the first is the plain equal-shape case.
SHAPE_PAIRS = [
    ((2, 3), (2, 3)),
    ((2, 3), (3,)),
    ((3,), (2, 3)),
    ((2, 1), (1, 3)),
    ((2, 3, 1), (3, 2)),
    ((3, 2), (2, 3, 1)),
    ((4,), ()),
    ((), (4,)),
]


@pytest.mark.parametrize("op", sorted(BINARY_OPS))
@pytest.mark.parametrize("shapes", SHAPE_PAIRS, ids=lambda s: f"{s[0]}x{s[1]}")
def test_binary_op_matches_torch(op, shapes, dtype):
    left_shape, right_shape = shapes
    # The base of a power must be positive for the exponent's gradient to exist.
    left = _positive(*left_shape)
    right = _positive(*right_shape) if op == "pow" else _signed(*right_shape)
    _check(BINARY_OPS[op], [left, right], dtype)


SCALAR_OPS = {
    "scalar_plus_tensor": lambda t: 2.0 + t,
    "scalar_minus_tensor": lambda t: 2.0 - t,
    "scalar_times_tensor": lambda t: 2.0 * t,
    "scalar_over_tensor": lambda t: 2.0 / t,
    "scalar_pow_tensor": lambda t: 2.0**t,
    "tensor_plus_scalar": lambda t: t + 2.0,
    "tensor_minus_scalar": lambda t: t - 2.0,
    "tensor_times_scalar": lambda t: t * 2.0,
    "tensor_over_scalar": lambda t: t / 2.0,
    "tensor_pow_scalar": lambda t: t**3.0,
}


@pytest.mark.parametrize("op", sorted(SCALAR_OPS))
def test_python_scalar_operand_matches_torch(op, dtype):
    _check(SCALAR_OPS[op], [_positive(2, 3)], dtype)


UNARY_OPS = {
    "neg": lambda t: -t,
    "exp": lambda t: t.exp(),
    "log": lambda t: t.log(),
}


@pytest.mark.parametrize("op", sorted(UNARY_OPS))
@pytest.mark.parametrize("shape", [(5,), (2, 3)], ids=str)
def test_unary_op_matches_torch(op, shape, dtype):
    _check(UNARY_OPS[op], [_positive(*shape)], dtype)


REDUCTIONS = {
    "sum_all": lambda t: t.sum(),
    "sum_dim0": lambda t: t.sum(0),
    "sum_dim1": lambda t: t.sum(1),
    "sum_dim_last": lambda t: t.sum(-1),
    "mean_all": lambda t: t.mean(),
    "mean_dim0": lambda t: t.mean(0),
    "mean_dim1": lambda t: t.mean(1),
    "mean_dim_last": lambda t: t.mean(-1),
}


@pytest.mark.parametrize("op", sorted(REDUCTIONS))
@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)], ids=str)
def test_reduction_matches_torch(op, shape, dtype):
    _check(REDUCTIONS[op], [_signed(*shape)], dtype)


@pytest.mark.parametrize("op", ["sum", "mean"])
@pytest.mark.parametrize("dim", [0, 1])
def test_reduction_keepdims_matches_torch(op, dim, dtype):
    # The keyword is ``keepdims`` in PyGrad and ``keepdim`` in torch.
    arr = _signed(2, 3)
    g_in, t_in = _pair(arr, dtype)
    g_out = getattr(g_in, op)(dim, keepdims=True)
    t_out = getattr(t_in, op)(dim, keepdim=True)
    _assert_same(g_out, t_out, dtype, "forward")

    upstream = _rng().uniform(0.5, 1.5, size=tuple(t_out.shape))
    g_up, t_up = _pair(upstream, dtype, requires_grad=False)
    g_out.backward(g_up)
    t_out.backward(t_up)
    _assert_same(g_in.grad, t_in.grad, dtype, "gradient")


def test_reused_node_accumulates_gradient_like_torch(dtype):
    # x is used three times and y twice, so gradients must be summed over paths.
    def fn(x, y):
        shared = x * y
        return (shared * x + shared / y - x).sum()

    _check(fn, [_positive(2, 3), _positive(2, 3)], dtype)


def test_composite_expression_matches_torch(dtype):
    def fn(a, b, c):
        return ((a * b).exp() / (c + 3.0) - (a**2.0).log() * b).mean()

    _check(fn, [_positive(3, 2), _positive(3, 2), _positive(2)], dtype)


def test_multi_layer_network_matches_torch(dtype):
    """Loss and every parameter gradient of a 4 -> 5 -> 4 -> 3 sigmoid network.

    ``matmul`` has no backward in PyGrad and views carry no gradient, so the
    network keeps activations as (features, batch) and writes each linear layer
    ``W @ h + b`` as ``(h * W[:, :, None]).sum(1) + b[:, None]`` with the weights
    stored directly as (fan_out, fan_in, 1) and the biases as (fan_out, 1). That
    is the same arithmetic as a matmul, and the identical expression is run in
    both libraries from identical inputs.
    """
    rng = _rng()
    batch, sizes = 6, [4, 5, 4, 3]
    x = rng.normal(size=(sizes[0], batch))
    target = rng.normal(size=(sizes[-1], batch))
    params, names = [], []
    for layer, (fan_in, fan_out) in enumerate(zip(sizes, sizes[1:]), start=1):
        params.append(rng.normal(scale=0.5, size=(fan_out, fan_in, 1)))
        params.append(rng.normal(scale=0.1, size=(fan_out, 1)))
        names += [f"W{layer}", f"b{layer}"]

    def network(h, y, *weights):
        layers = list(zip(weights[0::2], weights[1::2]))
        for i, (w, b) in enumerate(layers):
            h = (h * w).sum(1) + b
            if i < len(layers) - 1:
                h = 1.0 / (1.0 + (-h).exp())  # sigmoid
        return ((h - y) ** 2.0).mean()  # mean squared error

    g_x, t_x = _pair(x, dtype, requires_grad=False)
    g_y, t_y = _pair(target, dtype, requires_grad=False)
    pairs = [_pair(p, dtype) for p in params]

    g_loss = network(g_x, g_y, *[p[0] for p in pairs])
    t_loss = network(t_x, t_y, *[p[1] for p in pairs])
    _assert_same(g_loss, t_loss, dtype, "loss")

    g_loss.backward()
    t_loss.backward()
    for name, (g_p, t_p) in zip(names, pairs):
        assert g_p.grad is not None, f"{name}: PyGrad produced no gradient"
        _assert_same(g_p.grad, t_p.grad, dtype, f"gradient of {name}")


VIEWS = {
    "transpose": lambda t: t.transpose(0, 1),
    "permute": lambda t: t.permute(1, 0) if isinstance(t, torch.Tensor) else Tensor.permute(t, 1, 0),
    "reshape": lambda t: t.reshape(3, 2),
    "expand": lambda t: t.reshape(1, 2, 3).expand(4, 2, 3),
}
OPS_ON_VIEWS = {
    "add": lambda v, other: v + other,
    "mul": lambda v, other: other * v,
    "div": lambda v, other: v / other,
    "neg": lambda v, other: -v,
    "exp": lambda v, other: v.exp(),
    "sum": lambda v, other: v.sum(0),
    "mean": lambda v, other: v.mean(-1),
}


@pytest.mark.parametrize("op", sorted(OPS_ON_VIEWS))
@pytest.mark.parametrize("view", sorted(VIEWS))
def test_op_on_view_forward_matches_torch(view, op, dtype):
    """Ops must honour the strides of a view. Forward only: views carry no gradient."""
    g_base, t_base = _pair(_signed(2, 3), dtype, requires_grad=False)
    g_view, t_view = VIEWS[view](g_base), VIEWS[view](t_base)
    _assert_same(g_view, t_view, dtype, "view")

    g_other, t_other = _pair(_positive(*t_view.shape), dtype, requires_grad=False)
    _assert_same(
        OPS_ON_VIEWS[op](g_view, g_other), OPS_ON_VIEWS[op](t_view, t_other), dtype, "forward"
    )


MATMUL_SHAPES = [((4,), (4,)), ((2, 3), (3, 4)), ((2, 3), (3,)), ((3,), (3, 2))]


@pytest.mark.parametrize("shapes", MATMUL_SHAPES, ids=lambda s: f"{s[0]}@{s[1]}")
def test_matmul_forward_matches_torch(shapes, dtype):
    g_a, t_a = _pair(_signed(*shapes[0]), dtype, requires_grad=False)
    g_b, t_b = _pair(_positive(*shapes[1]), dtype, requires_grad=False)
    _assert_same(g_a @ g_b, t_a @ t_b, dtype, "matmul forward")


@pytest.mark.xfail(strict=True, reason="matmul has no backward implementation yet")
def test_matmul_gradient_matches_torch(dtype):
    _check(lambda a, b: a @ b, [_signed(2, 3), _positive(3, 4)], dtype)


@pytest.mark.xfail(strict=True, reason="views (reshape/transpose) do not propagate gradients yet")
@pytest.mark.parametrize("view", ["reshape", "transpose"])
def test_gradient_through_view_matches_torch(view, dtype):
    ops = {
        "reshape": lambda t: t.reshape(3, 2) * 2.0,
        "transpose": lambda t: t.transpose(0, 1) * 2.0,
    }
    _check(ops[view], [_signed(2, 3)], dtype)
