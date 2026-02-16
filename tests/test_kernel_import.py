import subprocess
import sys


def test_kernel_import_does_not_emit_warning_when_extension_is_available():
    proc = subprocess.run(
        [sys.executable, "-c", "import grad.kernels"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Could not import cpu_kernel extension" not in proc.stderr
