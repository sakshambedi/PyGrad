"""
Shared pretty-print utilities for pygrad example scripts.
"""


def print_section(title: str, width: int = 52, char: str = "=") -> None:
    """Print a consistently formatted section header."""
    line = char * width
    print(f"\n{line}")
    print(title)
    print(line)


def print_kv(label: str, value, label_width: int = 28) -> None:
    """Print an aligned key/value line."""
    print(f"{label:<{label_width}}: {value}")


def print_result(name: str, passed: bool, label_width: int = 28) -> None:
    """Print a standardized PASS/FAIL result line."""
    status = "PASSED" if passed else "FAILED"
    print_kv(f"Result [{name}]", status, label_width=label_width)


def print_banner(title: str, width: int = 52, char: str = "-") -> None:
    """Print a top-level banner for an example script."""
    print(title)
    print(char * width)


def print_summary(passed: int, total: int) -> None:
    """Print a standardized summary block."""
    print_section("Summary")
    print_kv("Demos passed", f"{passed}/{total}")
    print_kv("Overall status", "PASSED" if passed == total else "FAILED")
