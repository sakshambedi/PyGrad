#!/usr/bin/env python3
"""
Example script demonstrating how to use the pygrad C++ kernel.
"""

import sys

from examples.print_utils import (
    print_banner,
    print_kv,
    print_result,
    print_section,
    print_summary,
)
from grad.kernels import cpu_kernel


def demo_buffer_creation() -> bool:
    """Demonstrate basic buffer creation and operations."""
    print_section("Buffer Creation")

    float_buffer = cpu_kernel.Buffer([1.0, 2.0, 3.0, 4.0], "float32")
    int_buffer = cpu_kernel.Buffer([1, 2, 3, 4], "int32")

    print_kv("Float buffer", float_buffer)
    print_kv("Int buffer", int_buffer)
    print_kv("Float buffer[0]", float_buffer[0])
    print_kv("Int buffer[1]", int_buffer[1])
    print_kv("Float buffer size", float_buffer.size())
    print_kv("Float buffer dtype", float_buffer.get_dtype())

    return True


def demo_buffer_casting() -> bool:
    """Demonstrate buffer type casting."""
    print_section("Buffer Casting")

    original = cpu_kernel.Buffer([1.5, 2.7, 3.2], "float32")
    int_cast = original.cast("int32")
    double_cast = original.cast("float64")

    print_kv("Original buffer", original)
    print_kv("Cast to int32", int_cast)
    print_kv("Cast to float64", double_cast)

    return True


def demo_array_interface() -> bool:
    """Demonstrate NumPy array interface support."""
    print_section("Array Interface")

    try:
        import numpy as np

        buffer = cpu_kernel.Buffer([1, 2, 3, 4], "float32")
        interface = buffer.__array_interface__
        np_array = np.array(buffer)

        print_kv("Array interface", interface)
        print_kv("NumPy array", np_array)
        print_kv("NumPy dtype", np_array.dtype)

        return True
    except ImportError:
        print_kv("NumPy", "Not available, skipping this demo")
        return True


def demo_edge_cases() -> bool:
    """Demonstrate edge cases and basic error-prone scenarios."""
    print_section("Edge Cases")

    empty = cpu_kernel.Buffer([], "float32")
    single = cpu_kernel.Buffer([42], "int32")
    large = cpu_kernel.Buffer(list(range(100)), "int32")

    print_kv("Empty buffer", empty)
    print_kv("Empty buffer size", empty.size())
    print_kv("Single-element buffer", single)
    print_kv("Large buffer size", large.size())
    print_kv("Large buffer (truncated)", large)

    return True


def main() -> int:
    """Run all buffer demos and print a clean summary."""
    print_banner("pygrad C++ Kernel Examples")

    demos = [
        ("Buffer Creation", demo_buffer_creation),
        ("Buffer Casting", demo_buffer_casting),
        ("Array Interface", demo_array_interface),
        ("Edge Cases", demo_edge_cases),
    ]

    passed = 0
    total = len(demos)

    for name, demo in demos:
        try:
            result = demo()
            print_result(name, result)
            if result:
                passed += 1
        except Exception as exc:
            print_kv(f"Result [{name}]", f"FAILED ({exc})")

    print_summary(passed, total)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
