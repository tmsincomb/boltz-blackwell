#!/usr/bin/env python3
"""Generate manifest logs documenting hardware, installation, and validation."""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from boltz_blackwell import __version__

# Minimal test input for validation prediction
TEST_INPUT_YAML = """\
version: 1
sequences:
  - protein:
      id: test
      sequence: GGGGG
"""


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
class Manifest:
    """Manifest data structure."""

    generated_at: str = ""
    boltz_blackwell_version: str = __version__
    hardware: dict[str, Any] = field(default_factory=dict)
    software: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary for JSON serialization."""
        return {
            "generated_at": self.generated_at,
            "boltz_blackwell_version": self.boltz_blackwell_version,
            "hardware": self.hardware,
            "software": self.software,
            "validation": self.validation,
        }


def get_driver_version() -> Optional[str]:
    """Get NVIDIA driver version from nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def collect_hardware_info(manifest: Manifest) -> None:
    """Collect hardware information."""
    manifest.hardware["driver_version"] = get_driver_version()

    try:
        import torch

        if torch.cuda.is_available():
            manifest.hardware["gpu_name"] = torch.cuda.get_device_name(0)
            manifest.hardware["gpu_count"] = torch.cuda.device_count()
            cap = torch.cuda.get_device_capability(0)
            manifest.hardware["compute_capability"] = f"{cap[0]}.{cap[1]}"
            manifest.hardware["cuda_version"] = torch.version.cuda
        else:
            manifest.hardware["gpu_name"] = None
            manifest.hardware["gpu_count"] = 0
            manifest.hardware["compute_capability"] = None
            manifest.hardware["cuda_version"] = None
    except ImportError:
        manifest.hardware["gpu_name"] = None
        manifest.hardware["gpu_count"] = 0
        manifest.hardware["compute_capability"] = None
        manifest.hardware["cuda_version"] = None


def collect_software_info(manifest: Manifest) -> None:
    """Collect software version information."""
    manifest.software["python_version"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # PyTorch
    try:
        import torch

        manifest.software["pytorch_version"] = torch.__version__
        manifest.software["arch_list"] = torch.cuda.get_arch_list()
        arch_list = torch.cuda.get_arch_list()
        manifest.software["sm_120_support"] = (
            "sm_120" in arch_list or "compute_120" in arch_list
        )
    except ImportError:
        manifest.software["pytorch_version"] = None
        manifest.software["arch_list"] = []
        manifest.software["sm_120_support"] = False

    # Boltz
    try:
        import boltz

        manifest.software["boltz_version"] = getattr(boltz, "__version__", "unknown")
    except ImportError:
        manifest.software["boltz_version"] = None

    # PyTorch Lightning
    try:
        import pytorch_lightning

        manifest.software["pytorch_lightning_version"] = pytorch_lightning.__version__
    except ImportError:
        manifest.software["pytorch_lightning_version"] = None

    # NumPy
    try:
        import numpy

        manifest.software["numpy_version"] = numpy.__version__
    except ImportError:
        manifest.software["numpy_version"] = None

    # RDKit
    try:
        import rdkit

        manifest.software["rdkit_version"] = getattr(rdkit, "__version__", "unknown")
    except ImportError:
        manifest.software["rdkit_version"] = None


def collect_validation_info(manifest: Manifest, *, skip_prediction: bool) -> None:
    """Collect validation test results."""
    # CUDA availability
    try:
        import torch

        manifest.validation["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        manifest.validation["cuda_available"] = False

    # Tensor allocation test
    try:
        import torch

        if torch.cuda.is_available():
            _ = torch.rand(100, 100).cuda()
            manifest.validation["tensor_allocation"] = True
        else:
            manifest.validation["tensor_allocation"] = False
    except Exception:
        manifest.validation["tensor_allocation"] = False

    # Boltz CLI availability
    manifest.validation["boltz_cli_available"] = shutil.which("boltz") is not None

    # Prediction test
    if skip_prediction:
        manifest.validation["prediction_test"] = {"skipped": True}
    else:
        manifest.validation["prediction_test"] = run_validation_prediction()


def run_validation_prediction() -> dict[str, Any]:
    """
    Run a mini Boltz prediction to validate the installation.

    Returns:
        Dictionary with prediction test results
    """
    result: dict[str, Any] = {
        "passed": False,
        "input": "built-in test sequence (5 residues: GGGGG)",
    }

    if not shutil.which("boltz"):
        result["error"] = "boltz CLI not found"
        return result

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        input_file = tmppath / "test_input.yaml"
        output_dir = tmppath / "output"

        # Write test input
        input_file.write_text(TEST_INPUT_YAML)

        # Build command with fast settings
        cmd = [
            "boltz",
            "predict",
            str(input_file),
            "--out_dir",
            str(output_dir),
            "--diffusion_samples",
            "1",
            "--recycling_steps",
            "1",
            "--sampling_steps",
            "10",
            "--no_kernels",
        ]

        start_time = time.time()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            duration = time.time() - start_time
            result["duration_seconds"] = round(duration, 2)

            if proc.returncode == 0:
                result["passed"] = True

                # Try to extract confidence score
                confidence_file = list(output_dir.rglob("confidence_*.json"))
                if confidence_file:
                    try:
                        confidence_data = json.loads(confidence_file[0].read_text())
                        result["confidence_score"] = confidence_data.get(
                            "confidence_score"
                        )
                    except (json.JSONDecodeError, OSError):
                        pass
            else:
                result["error"] = proc.stderr[:500] if proc.stderr else "Unknown error"

        except subprocess.TimeoutExpired:
            result["error"] = "Prediction timed out after 5 minutes"
        except Exception as e:
            result["error"] = str(e)

    return result


def generate_manifest(*, skip_validation: bool = False) -> Manifest:
    """
    Generate a complete manifest.

    Args:
        skip_validation: Skip the prediction validation test

    Returns:
        Populated Manifest instance
    """
    manifest = Manifest()
    manifest.generated_at = datetime.now(timezone.utc).isoformat()

    collect_hardware_info(manifest)
    collect_software_info(manifest)
    collect_validation_info(manifest, skip_prediction=skip_validation)

    return manifest


def print_progress(message: str) -> None:
    """Print a progress message to stderr."""
    print(f"{COLORS.BLUE}➤ {message}{COLORS.RESET}", file=sys.stderr)


def main(args=None) -> int:
    """CLI entry point for boltz-blackwell-manifest."""
    if args is None:
        import argparse

        parser = argparse.ArgumentParser(
            description="Generate manifest documenting hardware, software, and validation"
        )
        parser.add_argument(
            "--skip-validation",
            action="store_true",
            help="Skip the prediction validation test (faster)",
        )
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            default=None,
            help="Output file path (default: stdout)",
        )
        args = parser.parse_args()

    skip_validation = getattr(args, "skip_validation", False)
    output_file = getattr(args, "output", None)

    # Show progress to stderr so it doesn't interfere with JSON output
    print_progress("Collecting hardware information...")
    print_progress("Collecting software versions...")

    if not skip_validation:
        print_progress("Running validation prediction (this may take a minute)...")

    manifest = generate_manifest(skip_validation=skip_validation)
    json_output = json.dumps(manifest.to_dict(), indent=2)

    if output_file:
        Path(output_file).write_text(json_output)
        print_progress(f"Manifest saved to: {output_file}")
    else:
        print(json_output)

    # Return success if validation passed or was skipped
    validation = manifest.validation.get("prediction_test", {})
    if validation.get("skipped") or validation.get("passed"):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
