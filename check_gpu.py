#!/usr/bin/env python3
"""
Check GPU compatibility for Boltz on Blackwell GPUs.
"""

import sys


def main():
    print("=" * 60)
    print("GPU Compatibility Check for Boltz on Blackwell")
    print("=" * 60)

    errors = []

    # Check PyTorch
    print("\n[PyTorch]")
    try:
        import torch

        print(f"  Version: {torch.__version__}")
        print(f"  CUDA Version: {torch.version.cuda}")

        if torch.cuda.is_available():
            print(f"  CUDA Available: Yes")
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
                errors.append(
                    "sm_120 not in arch list - need PyTorch nightly with CUDA 13.0"
                )
                print("  ✗ sm_120 NOT available - Blackwell may not work")

            # Test tensor allocation
            try:
                x = torch.rand(100, 100).cuda()
                print(f"  ✓ GPU tensor allocation works")
            except Exception as e:
                errors.append(f"GPU tensor allocation failed: {e}")
                print(f"  ✗ GPU tensor allocation failed: {e}")
        else:
            errors.append("CUDA not available")
            print("  ✗ CUDA NOT available")

    except ImportError:
        errors.append("PyTorch not installed")
        print("  ✗ PyTorch not installed")

    # Check Boltz
    print("\n[Boltz]")
    try:
        import boltz

        version = getattr(boltz, "__version__", "unknown")
        print(f"  Version: {version}")
        print("  ✓ Boltz installed")
    except ImportError:
        errors.append("Boltz not installed")
        print("  ✗ Boltz not installed")

    # Check other dependencies
    print("\n[Dependencies]")
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

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("Issues Found:")
        for error in errors:
            print(f"  ✗ {error}")
        print("\nRecommended fix:")
        print(
            "  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130"
        )
        print("  pip install boltz[cuda] -U")
        return 1
    else:
        print("✓ All checks passed! Boltz should work on this GPU.")
        print("\nRun a prediction with:")
        print("  boltz predict examples/insulin_a.yaml --out_dir output/ --no_kernels")
        return 0


if __name__ == "__main__":
    sys.exit(main())
