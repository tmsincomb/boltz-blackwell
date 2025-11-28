#!/usr/bin/env python3
"""
Installer for Boltz on Blackwell GPUs.

Installs PyTorch nightly with CUDA 13.0 and Boltz with CUDA support.
"""

import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Colors:
    """ANSI color codes for terminal output."""

    RED: str = "\033[0;31m"
    GREEN: str = "\033[0;32m"
    YELLOW: str = "\033[1;33m"
    BLUE: str = "\033[0;34m"
    RESET: str = "\033[0m"


COLORS = Colors()


def print_header(text: str) -> None:
    """Print a styled header."""
    separator = "=" * 60
    print(f"\n{COLORS.BLUE}{separator}{COLORS.RESET}")
    print(f"{COLORS.BLUE}{text}{COLORS.RESET}")
    print(f"{COLORS.BLUE}{separator}{COLORS.RESET}")


def print_step(text: str) -> None:
    """Print a step message."""
    print(f"\n{COLORS.YELLOW}➤ {text}{COLORS.RESET}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{COLORS.GREEN}✓ {text}{COLORS.RESET}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{COLORS.RED}✗ {text}{COLORS.RESET}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{COLORS.YELLOW}⚠ {text}{COLORS.RESET}")


def run_command(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    quiet: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a command and handle errors.

    Args:
        cmd: Command and arguments to run
        check: Raise exception on non-zero exit code
        capture: Capture stdout and stderr
        quiet: Suppress command echo

    Returns:
        CompletedProcess instance
    """
    if not quiet:
        print(f"  Running: {' '.join(cmd)}")

    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)

    return subprocess.run(cmd, check=check)


def check_nvidia_gpu() -> tuple[bool, Optional[str], Optional[str]]:
    """
    Check for NVIDIA GPU.

    Returns:
        Tuple of (found, gpu_name, compute_capability)
    """
    if not shutil.which("nvidia-smi"):
        return False, None, None

    try:
        result = run_command(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture=True,
            quiet=True,
        )
        parts = result.stdout.strip().split(", ")
        name = parts[0] if parts else "Unknown"
        compute_cap = parts[1] if len(parts) >= 2 else "Unknown"
        return True, name, compute_cap
    except subprocess.CalledProcessError:
        return False, None, None


def check_python_version() -> tuple[bool, str]:
    """
    Check Python version meets requirements.

    Returns:
        Tuple of (is_valid, version_string)
    """
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    is_valid = version >= (3, 10)
    return is_valid, version_str


def install_pytorch(*, quiet: bool = False) -> bool:
    """
    Install PyTorch nightly with CUDA 13.0.

    Args:
        quiet: Minimal output

    Returns:
        True on success, False on failure
    """
    print_step("Installing PyTorch nightly with CUDA 13.0...")
    print("  This provides sm_120 support with forward compatibility for sm_121 (Blackwell)")

    try:
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--pre",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/nightly/cu130",
                "--upgrade",
            ],
            quiet=quiet,
        )
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install PyTorch: {e}")
        return False


def verify_pytorch() -> bool:
    """
    Verify PyTorch installation and CUDA availability.

    Returns:
        True if verification passes
    """
    print_step("Verifying PyTorch installation...")

    try:
        import torch
    except ImportError:
        print_error("PyTorch not installed correctly")
        return False

    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print_error("CUDA not available in PyTorch")
        return False

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    arch_list = torch.cuda.get_arch_list()
    print(f"  Arch List: {arch_list}")

    if "sm_120" in arch_list:
        print_success("sm_120 support available (Blackwell compatible)")
    else:
        print_warning("sm_120 not in arch list - may have compatibility issues")

    return True


def install_boltz(*, quiet: bool = False) -> bool:
    """
    Install Boltz with CUDA support.

    Args:
        quiet: Minimal output

    Returns:
        True on success, False on failure
    """
    print_step("Installing Boltz with CUDA support...")

    try:
        run_command(
            [sys.executable, "-m", "pip", "install", "boltz[cuda]", "-U"],
            quiet=quiet,
        )
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install Boltz: {e}")
        return False


def verify_boltz() -> bool:
    """
    Verify Boltz installation.

    Returns:
        True if Boltz imports successfully
    """
    print_step("Verifying Boltz installation...")

    try:
        import boltz

        version = getattr(boltz, "__version__", "unknown")
        print(f"  Boltz Version: {version}")
        print_success("Boltz installed successfully")
        return True
    except ImportError as e:
        print_error(f"Boltz import failed: {e}")
        return False


def check_boltz_cli() -> bool:
    """
    Check if boltz CLI is available in PATH.

    Returns:
        True if CLI is found
    """
    if shutil.which("boltz"):
        print_success("Boltz CLI is available")
        return True

    print_warning("Boltz CLI not found in PATH")
    return False


def handle_gpu_check(force: bool) -> bool:
    """
    Handle GPU detection and compatibility check.

    Args:
        force: Continue even if GPU check fails

    Returns:
        True if should continue with installation
    """
    print_step("Checking GPU...")
    gpu_found, gpu_name, compute_cap = check_nvidia_gpu()

    if not gpu_found:
        print_error("nvidia-smi not found. Please install NVIDIA drivers.")
        if not force:
            return False
        print_warning("Continuing anyway due to --force flag")
        return True

    print(f"  Found GPU: {gpu_name} (Compute Capability: {compute_cap})")

    if compute_cap == "12.1":
        return True

    print_warning("This script is optimized for Blackwell GPUs (compute capability 12.1)")
    print_warning(f"Your GPU has compute capability {compute_cap}")

    if force:
        return True

    response = input("Continue anyway? (y/n) ").strip().lower()
    return response == "y"


def print_success_message() -> None:
    """Print installation success message with usage instructions."""
    print_header("Installation Complete!")
    print(f"""
To run a prediction:
  boltz predict examples/insulin_a.yaml --out_dir output/ --no_kernels

Or use the boltz-blackwell wrapper:
  boltz-blackwell predict examples/insulin_a.yaml

To verify installation:
  boltz-blackwell check

{COLORS.YELLOW}IMPORTANT: Always use --no_kernels flag on Blackwell GPUs{COLORS.RESET}
""")


def install(
    skip_pytorch: bool = False,
    skip_boltz: bool = False,
    force: bool = False,
    quiet: bool = False,
) -> int:
    """
    Main installation function.

    Args:
        skip_pytorch: Skip PyTorch installation
        skip_boltz: Skip Boltz installation
        force: Force installation even if GPU check fails
        quiet: Minimal output

    Returns:
        0 on success, 1 on failure
    """
    print_header("Boltz Installation for Blackwell GPUs")

    # Check GPU
    if not handle_gpu_check(force):
        return 1

    # Check Python
    print_step("Checking Python...")
    python_ok, python_version = check_python_version()
    print(f"  Python {python_version}")

    if not python_ok:
        print_error("Python 3.10+ required")
        return 1
    print_success("Python version OK")

    # Install PyTorch
    if skip_pytorch:
        print_step("Skipping PyTorch installation (--skip-pytorch)")
    else:
        if not install_pytorch(quiet=quiet):
            return 1

        # Reload torch after installation
        if "torch" in sys.modules:
            del sys.modules["torch"]

        if not verify_pytorch():
            print_warning("PyTorch verification had issues, but continuing...")

    # Install Boltz
    if skip_boltz:
        print_step("Skipping Boltz installation (--skip-boltz)")
    else:
        if not install_boltz(quiet=quiet):
            return 1

        # Reload boltz after installation
        if "boltz" in sys.modules:
            del sys.modules["boltz"]

        if not verify_boltz():
            return 1

        check_boltz_cli()

    print_success_message()
    return 0


def main(args=None) -> int:
    """CLI entry point for boltz-blackwell-install."""
    if args is not None:
        return install(
            skip_pytorch=getattr(args, "skip_pytorch", False),
            skip_boltz=getattr(args, "skip_boltz", False),
            force=getattr(args, "force", False),
            quiet=getattr(args, "quiet", False),
        )

    import argparse

    parser = argparse.ArgumentParser(
        description="Install PyTorch and Boltz for Blackwell GPUs"
    )
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Skip PyTorch installation",
    )
    parser.add_argument(
        "--skip-boltz",
        action="store_true",
        help="Skip Boltz installation",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force installation even if GPU check fails",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output",
    )
    parsed_args = parser.parse_args()

    return install(
        skip_pytorch=parsed_args.skip_pytorch,
        skip_boltz=parsed_args.skip_boltz,
        force=parsed_args.force,
        quiet=parsed_args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
