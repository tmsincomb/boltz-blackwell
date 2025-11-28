#!/usr/bin/env python3
"""
Wrapper for boltz predict with Blackwell GPU optimizations.

Automatically adds --no_kernels flag and provides convenient presets.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
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


def predict(
    input_file: str,
    *,
    output_dir: str = "output",
    samples: int = 1,
    recycling: int = 3,
    steps: int = 200,
    use_msa_server: bool = False,
    extra_args: Optional[list[str]] = None,
) -> int:
    """
    Run Boltz prediction with Blackwell-optimized settings.

    Args:
        input_file: Path to input YAML file
        output_dir: Output directory
        samples: Number of diffusion samples
        recycling: Number of recycling steps
        steps: Number of sampling steps
        use_msa_server: Whether to use ColabFold MSA server
        extra_args: Additional arguments to pass to boltz

    Returns:
        Exit code from boltz command
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"{COLORS.RED}Error: Input file not found: {input_file}{COLORS.RESET}")
        return 1

    cmd = [
        "boltz",
        "predict",
        str(input_path),
        "--out_dir",
        output_dir,
        "--diffusion_samples",
        str(samples),
        "--recycling_steps",
        str(recycling),
        "--sampling_steps",
        str(steps),
        "--no_kernels",  # Required for Blackwell GPUs
    ]

    if use_msa_server:
        cmd.append("--use_msa_server")

    if extra_args:
        cmd.extend(extra_args)

    print(f"{COLORS.BLUE}Running Boltz prediction...{COLORS.RESET}")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_dir}")
    print(f"  Samples: {samples}")
    print(f"  Recycling: {recycling}")
    print(f"  Steps: {steps}")
    if use_msa_server:
        print("  MSA Server: enabled")
    print()

    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print()
            print(f"{COLORS.GREEN}Done! Results saved to: {output_dir}{COLORS.RESET}")
        return result.returncode
    except FileNotFoundError:
        print(
            f"{COLORS.RED}Error: boltz command not found. "
            f"Run 'boltz-blackwell install' first.{COLORS.RESET}"
        )
        return 1
    except KeyboardInterrupt:
        print(f"\n{COLORS.YELLOW}Prediction cancelled.{COLORS.RESET}")
        return 130


@dataclass
class PredictSettings:
    """Settings for a prediction run."""

    samples: int
    recycling: int
    steps: int


PRESET_FAST = PredictSettings(samples=1, recycling=1, steps=10)
PRESET_HIGH_QUALITY = PredictSettings(samples=5, recycling=3, steps=200)


def main(args=None) -> int:
    """CLI entry point for boltz-blackwell-predict."""
    if args is not None:
        # Apply presets
        if getattr(args, "fast", False):
            settings = PRESET_FAST
        elif getattr(args, "high_quality", False):
            settings = PRESET_HIGH_QUALITY
        else:
            settings = PredictSettings(
                samples=args.samples,
                recycling=args.recycling,
                steps=args.steps,
            )

        return predict(
            input_file=args.input,
            output_dir=getattr(args, "output", "output"),
            samples=settings.samples,
            recycling=settings.recycling,
            steps=settings.steps,
            use_msa_server=getattr(args, "use_msa_server", False),
        )

    import argparse

    parser = argparse.ArgumentParser(
        description="Run Boltz prediction with Blackwell-optimized settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  boltz-blackwell-predict input.yaml
  boltz-blackwell-predict input.yaml --fast
  boltz-blackwell-predict input.yaml --high-quality -o results/
  boltz-blackwell-predict input.yaml --use-msa-server
""",
    )
    parser.add_argument(
        "input",
        help="Input YAML file describing the protein/complex",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=1,
        help="Number of diffusion samples (default: 1)",
    )
    parser.add_argument(
        "--recycling",
        "-r",
        type=int,
        default=3,
        help="Number of recycling steps (default: 3)",
    )
    parser.add_argument(
        "--steps",
        "-t",
        type=int,
        default=200,
        help="Number of sampling steps (default: 200)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: 1 sample, 1 recycling, 10 steps",
    )
    parser.add_argument(
        "--high-quality",
        action="store_true",
        help="High quality: 5 samples, 3 recycling, 200 steps",
    )
    parser.add_argument(
        "--use-msa-server",
        action="store_true",
        help="Use ColabFold MSA server (requires internet)",
    )
    parsed_args = parser.parse_args()

    # Apply presets
    if parsed_args.fast:
        settings = PRESET_FAST
    elif parsed_args.high_quality:
        settings = PRESET_HIGH_QUALITY
    else:
        settings = PredictSettings(
            samples=parsed_args.samples,
            recycling=parsed_args.recycling,
            steps=parsed_args.steps,
        )

    return predict(
        input_file=parsed_args.input,
        output_dir=parsed_args.output,
        samples=settings.samples,
        recycling=settings.recycling,
        steps=settings.steps,
        use_msa_server=parsed_args.use_msa_server,
    )


if __name__ == "__main__":
    sys.exit(main())
