#!/usr/bin/env python3
"""
Example script demonstrating how to use the pygrad C++ kernel
"""

import sys

try:
    from grad.kernels import cpu_kernel  # type: ignore

    print("✅ Successfully imported cpu_kernel module")
except ImportError as e:
    print(f"❌ Failed to import cpu_kernel: {e}")
    print("Make sure to build the project first:")
    print("  ./build.sh --debug --tests")
    sys.exit(1)


def test_buffer_creation():
    """Test basic buffer creation and operations"""
    print("\n=== Testing Buffer Creation ===")

    # Create buffers with different data types
    float_buffer = cpu_kernel.Buffer([1.0, 2.0, 3.0, 4.0], "float32")
    int_buffer = cpu_kernel.Buffer([1, 2, 3, 4], "int32")

    print(f"Float buffer: {float_buffer}")
    print(f"Int buffer: {int_buffer}")

    # Test accessing elements
    print(f"Float buffer[0]: {float_buffer[0]}")
    print(f"Int buffer[1]: {int_buffer[1]}")

    # Test buffer properties
    print(f"Float buffer size: {float_buffer.size()}")
    print(f"Float buffer dtype: {float_buffer.get_dtype()}")

    return True


def test_buffer_casting():
    """Test buffer type casting"""
    print("\n=== Testing Buffer Casting ===")

    # Create a float buffer
    original = cpu_kernel.Buffer([1.5, 2.7, 3.2], "float32")
    print(f"Original buffer: {original}")

    # Cast to different types
    int_cast = original.cast("int32")
    double_cast = original.cast("float64")

    print(f"Cast to int32: {int_cast}")
    print(f"Cast to float64: {double_cast}")

    return True


def test_array_interface():
    """Test NumPy array interface"""
    print("\n=== Testing Array Interface ===")

    try:
        import numpy as np

        # Create a buffer
        buffer = cpu_kernel.Buffer([1, 2, 3, 4], "float32")

        # Get array interface
        interface = buffer.__array_interface__
        print(f"Array interface: {interface}")

        # Convert to NumPy array
        np_array = np.array(buffer)
        print(f"NumPy array: {np_array}")
        print(f"NumPy array dtype: {np_array.dtype}")

        return True
    except ImportError:
        print("NumPy not available, skipping array interface test")
        return True


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n=== Testing Edge Cases ===")

    # Test empty buffer
    empty = cpu_kernel.Buffer([], "float32")
    print(f"Empty buffer: {empty}")
    print(f"Empty buffer size: {empty.size()}")

    # Test single element
    single = cpu_kernel.Buffer([42], "int32")
    print(f"Single element buffer: {single}")

    # Test large buffer
    large_data = list(range(100))
    large = cpu_kernel.Buffer(large_data, "int32")
    print(f"Large buffer size: {large.size()}")
    print(f"Large buffer (truncated): {large}")

    return True


def main():
    """Main test function"""
    print("pygrad C++ Kernel Example")
    print("============================")

    tests = [
        test_buffer_creation,
        test_buffer_casting,
        test_array_interface,
        test_edge_cases,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ Test passed")
            else:
                print("❌ Test failed")
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")

    print("\n=== Summary ===")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
