import sys

try:
    from . import cpu_kernel
except ImportError:
    print(
        "Warning: Could not import cpu_kernel extension. Make sure it's properly compiled.",
        file=sys.stderr,
    )
    raise

__all__ = ["cpu_kernel"]
