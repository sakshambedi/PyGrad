import sys

from grad.kernels import cpu_kernel

__all__ = ["cpu_kernel"]

print(
    "Warning: Could not import cpu_kernel extension. Make sure it's properly compiled.",
    file=sys.stderr,
)
