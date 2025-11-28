#!/usr/bin/env python3
"""Check GPU compatibility for Boltz on Blackwell GPUs."""

import sys
from typing import Optional

SEPARATOR_WIDTH = 60


def check_pytorch() -> tuple[bool, list[str]]:
    """
    Check PyTorch installation and CUDA availability.

    Returns:
        Tuple of (success, list of error messages)
    """
    errors: list[str] = []

    try:
        import torch
    except ImportError:
        return False, ["PyTorch not installed"]

    print(f"  Version: {torch.__version__}")
    print(f"  CUDA Version: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("  CUDA Available: No")
        return False, ["CUDA not available"]

    print("  CUDA Available: Yes")
    print(f"  GPU Count: {torch.cuda.device_count()}")
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")

    cap = torch.cuda.get_device_capability(0)
    print(f"  Compute Capability: {cap[0]}.{cap[1]}")

    arch_list = torch.cuda.get_arch_list()
    print(f"  Supported Architectures: {arch_list}")

    # Check for sm_120 (needed for sm_121 forward compatibility)
    if "sm_120" in arch_list or "compute_120" in arch_list:
        print("  ✓ sm_120 support available (Blackwell compatible)")
    else:
        errors.append("sm_120 not in arch list - need PyTorch nightly with CUDA 13.0")
        print("  ✗ sm_120 NOT available - Blackwell may not work")

    # Test tensor allocation
    try:
        _ = torch.rand(100, 100).cuda()
        print("  ✓ GPU tensor allocation works")
    except Exception as e:
        errors.append(f"GPU tensor allocation failed: {e}")
        print(f"  ✗ GPU tensor allocation failed: {e}")

    return len(errors) == 0, errors


def check_boltz() -> tuple[bool, Optional[str]]:
    """
    Check Boltz installation.

    Returns:
        Tuple of (success, error message if any)
    """
    try:
        import boltz

        version = getattr(boltz, "__version__", "unknown")
        print(f"  Version: {version}")
        print("  ✓ Boltz installed")
        return True, None
    except ImportError:
        print("  ✗ Boltz not installed")
        return False, "Boltz not installed"


def check_dependencies() -> list[str]:
    """
    Check other required dependencies.

    Returns:
        List of error messages for missing dependencies
    """
    errors: list[str] = []
    deps = [
        ("pytorch_lightning", "PyTorch Lightning"),
        ("rdkit", "RDKit"),
        ("numpy", "NumPy"),
    ]

    for module, name in deps:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {name}: {version}")
        except ImportError:
            errors.append(f"{name} not installed")
            print(f"  ✗ {name} not installed")

    return errors


def print_summary(errors: list[str]) -> None:
    """Print summary and recommendations based on errors found."""
    print("\n" + "=" * SEPARATOR_WIDTH)

    if not errors:
        print("✓ All checks passed! Boltz should work on this GPU.")
        print("\nRun a prediction with:")
        print("  boltz predict examples/insulin_a.yaml --out_dir output/ --no_kernels")
        return

    print("Issues Found:")
    for error in errors:
        print(f"  ✗ {error}")

    print("\nRecommended fix:")
    print(
        "  pip install --pre torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/nightly/cu130"
    )
    print("  pip install boltz[cuda] -U")


def main() -> int:
    """
    Run GPU compatibility checks for Boltz on Blackwell.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("=" * SEPARATOR_WIDTH)
    print("GPU Compatibility Check for Boltz on Blackwell")
    print("=" * SEPARATOR_WIDTH)

    errors: list[str] = []

    # Check PyTorch
    print("\n[PyTorch]")
    _, pytorch_errors = check_pytorch()
    errors.extend(pytorch_errors)

    # Check Boltz
    print("\n[Boltz]")
    _, boltz_error = check_boltz()
    if boltz_error:
        errors.append(boltz_error)

    # Check dependencies
    print("\n[Dependencies]")
    dep_errors = check_dependencies()
    errors.extend(dep_errors)

    # Summary
    print_summary(errors)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
