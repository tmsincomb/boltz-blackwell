#!/usr/bin/env python3
"""Check GPU compatibility for Boltz on Blackwell GPUs."""

import json
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Colors:
    """ANSI color codes for terminal output."""

    RED: str = "\033[0;31m"
    GREEN: str = "\033[0;32m"
    YELLOW: str = "\033[1;33m"
    BLUE: str = "\033[0;34m"
    RESET: str = "\033[0m"


COLORS = Colors()


@dataclass
class CheckResults:
    """Results from GPU compatibility check."""

    success: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pytorch: dict[str, Any] = field(default_factory=dict)
    boltz_info: dict[str, Any] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "errors": self.errors,
            "warnings": self.warnings,
            "pytorch": self.pytorch,
            "boltz": self.boltz_info,
            "gpu": self.gpu,
            "dependencies": self.dependencies,
        }


def check_pytorch(results: CheckResults) -> None:
    """
    Check PyTorch installation and update results.

    Args:
        results: CheckResults instance to update
    """
    try:
        import torch
    except ImportError:
        results.pytorch["installed"] = False
        results.errors.append("PyTorch not installed")
        results.success = False
        return

    results.pytorch["installed"] = True
    results.pytorch["version"] = torch.__version__
    results.pytorch["cuda_version"] = torch.version.cuda
    results.pytorch["cuda_available"] = torch.cuda.is_available()

    if not torch.cuda.is_available():
        results.errors.append("CUDA not available")
        results.success = False
        return

    results.gpu["count"] = torch.cuda.device_count()
    results.gpu["name"] = torch.cuda.get_device_name(0)

    cap = torch.cuda.get_device_capability(0)
    results.gpu["compute_capability"] = f"{cap[0]}.{cap[1]}"

    arch_list = torch.cuda.get_arch_list()
    results.pytorch["arch_list"] = arch_list

    has_sm120 = "sm_120" in arch_list or "compute_120" in arch_list
    results.pytorch["sm_120_support"] = has_sm120

    if not has_sm120:
        results.errors.append(
            "sm_120 not in arch list - need PyTorch nightly with CUDA 13.0"
        )
        results.success = False

    # Test tensor allocation
    try:
        _ = torch.rand(100, 100).cuda()
        results.gpu["tensor_allocation"] = True
    except Exception as e:
        results.errors.append(f"GPU tensor allocation failed: {e}")
        results.gpu["tensor_allocation"] = False
        results.success = False


def check_boltz(results: CheckResults) -> None:
    """
    Check Boltz installation and update results.

    Args:
        results: CheckResults instance to update
    """
    try:
        import boltz

        results.boltz_info["installed"] = True
        results.boltz_info["version"] = getattr(boltz, "__version__", "unknown")
    except ImportError:
        results.boltz_info["installed"] = False
        results.errors.append("Boltz not installed")
        results.success = False

    results.boltz_info["cli_available"] = shutil.which("boltz") is not None


def check_dependencies(results: CheckResults) -> None:
    """
    Check required dependencies and update results.

    Args:
        results: CheckResults instance to update
    """
    deps = [
        ("pytorch_lightning", "PyTorch Lightning"),
        ("rdkit", "RDKit"),
        ("numpy", "NumPy"),
    ]

    for module, name in deps:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            results.dependencies[name] = {"installed": True, "version": version}
        except ImportError:
            results.dependencies[name] = {"installed": False}
            results.warnings.append(f"{name} not installed")


def check_gpu() -> dict[str, Any]:
    """
    Check GPU compatibility and return results.

    Returns:
        Dictionary with check results
    """
    results = CheckResults()

    check_pytorch(results)
    check_boltz(results)
    check_dependencies(results)

    return results.to_dict()


def print_pytorch_section(results: dict[str, Any]) -> None:
    """Print PyTorch check results."""
    print(f"\n{COLORS.YELLOW}[PyTorch]{COLORS.RESET}")

    pt = results.get("pytorch", {})
    if not pt.get("installed"):
        print(f"  {COLORS.RED}✗ PyTorch not installed{COLORS.RESET}")
        return

    print(f"  Version: {pt.get('version', 'unknown')}")
    print(f"  CUDA Version: {pt.get('cuda_version', 'unknown')}")
    cuda_status = "Yes" if pt.get("cuda_available") else "No"
    print(f"  CUDA Available: {cuda_status}")

    if not pt.get("cuda_available"):
        return

    gpu = results.get("gpu", {})
    print(f"  GPU Count: {gpu.get('count', 'unknown')}")
    print(f"  GPU Name: {gpu.get('name', 'unknown')}")
    print(f"  Compute Capability: {gpu.get('compute_capability', 'unknown')}")
    print(f"  Supported Architectures: {pt.get('arch_list', [])}")

    if pt.get("sm_120_support"):
        print(f"  {COLORS.GREEN}✓ sm_120 support available (Blackwell compatible){COLORS.RESET}")
    else:
        print(f"  {COLORS.RED}✗ sm_120 NOT available - Blackwell may not work{COLORS.RESET}")

    if gpu.get("tensor_allocation"):
        print(f"  {COLORS.GREEN}✓ GPU tensor allocation works{COLORS.RESET}")
    else:
        print(f"  {COLORS.RED}✗ GPU tensor allocation failed{COLORS.RESET}")


def print_boltz_section(results: dict[str, Any]) -> None:
    """Print Boltz check results."""
    print(f"\n{COLORS.YELLOW}[Boltz]{COLORS.RESET}")

    boltz_info = results.get("boltz", {})
    if not boltz_info.get("installed"):
        print(f"  {COLORS.RED}✗ Boltz not installed{COLORS.RESET}")
        return

    print(f"  Version: {boltz_info.get('version', 'unknown')}")
    print(f"  {COLORS.GREEN}✓ Boltz installed{COLORS.RESET}")

    if boltz_info.get("cli_available"):
        print(f"  {COLORS.GREEN}✓ Boltz CLI available{COLORS.RESET}")
    else:
        print(f"  {COLORS.RED}✗ Boltz CLI not in PATH{COLORS.RESET}")


def print_dependencies_section(results: dict[str, Any]) -> None:
    """Print dependencies check results."""
    print(f"\n{COLORS.YELLOW}[Dependencies]{COLORS.RESET}")

    for name, info in results.get("dependencies", {}).items():
        if info.get("installed"):
            print(f"  {COLORS.GREEN}✓ {name}: {info.get('version', 'unknown')}{COLORS.RESET}")
        else:
            print(f"  {COLORS.RED}✗ {name} not installed{COLORS.RESET}")


def print_summary(results: dict[str, Any]) -> None:
    """Print summary and recommendations."""
    print("\n" + "=" * 60)

    errors = results.get("errors", [])
    warnings = results.get("warnings", [])

    if errors:
        print(f"{COLORS.RED}Issues Found:{COLORS.RESET}")
        for error in errors:
            print(f"  {COLORS.RED}✗ {error}{COLORS.RESET}")

    if warnings:
        print(f"{COLORS.YELLOW}Warnings:{COLORS.RESET}")
        for warning in warnings:
            print(f"  {COLORS.YELLOW}⚠ {warning}{COLORS.RESET}")

    if results.get("success"):
        print(f"\n{COLORS.GREEN}✓ All checks passed! Boltz should work on this GPU.{COLORS.RESET}")
        print("\nRun a prediction with:")
        print("  boltz predict examples/insulin_a.yaml --out_dir output/ --no_kernels")
        print("\nOr use the boltz-blackwell wrapper:")
        print("  boltz-blackwell predict examples/insulin_a.yaml")
    else:
        print(f"\n{COLORS.RED}✗ Some checks failed.{COLORS.RESET}")
        print(f"\n{COLORS.YELLOW}Recommended fix:{COLORS.RESET}")
        print("  boltz-blackwell-install")
        print("\nOr manually:")
        print(
            "  pip install --pre torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/nightly/cu130"
        )
        print("  pip install boltz[cuda] -U")


def print_results(results: dict[str, Any]) -> None:
    """Print check results in a human-readable format."""
    print(f"\n{COLORS.BLUE}{'=' * 60}{COLORS.RESET}")
    print(f"{COLORS.BLUE}GPU Compatibility Check for Boltz on Blackwell{COLORS.RESET}")
    print(f"{COLORS.BLUE}{'=' * 60}{COLORS.RESET}")

    print_pytorch_section(results)
    print_boltz_section(results)
    print_dependencies_section(results)
    print_summary(results)


def main(args=None) -> int:
    """CLI entry point for boltz-blackwell-check."""
    if args is None:
        import argparse

        parser = argparse.ArgumentParser(
            description="Check GPU compatibility for Boltz on Blackwell"
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output results as JSON",
        )
        args = parser.parse_args()

    results = check_gpu()

    if getattr(args, "json", False):
        print(json.dumps(results, indent=2))
    else:
        print_results(results)

    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
