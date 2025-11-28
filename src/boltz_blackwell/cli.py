#!/usr/bin/env python3
"""Main CLI entry point for boltz-blackwell."""

import argparse
import sys
from typing import NoReturn

from boltz_blackwell import __version__


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="boltz-blackwell",
        description="Boltz installation and utilities for NVIDIA Blackwell GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  install     Install PyTorch and Boltz with Blackwell GPU support
  check       Check GPU compatibility and installation status
  predict     Run Boltz prediction with Blackwell-optimized settings
  manifest    Generate manifest documenting hardware and validation

Examples:
  boltz-blackwell install              # Install everything
  boltz-blackwell check                # Verify installation
  boltz-blackwell predict input.yaml   # Run prediction
  boltz-blackwell manifest             # Generate manifest JSON

For more information on each command:
  boltz-blackwell <command> --help
""",
    )

    parser.add_argument(
        "--version", "-V", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _add_install_subparser(subparsers)
    _add_check_subparser(subparsers)
    _add_predict_subparser(subparsers)
    _add_manifest_subparser(subparsers)

    return parser


def _add_install_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add the install subcommand."""
    install_parser = subparsers.add_parser(
        "install",
        help="Install PyTorch and Boltz with Blackwell GPU support",
        description="Install PyTorch nightly with CUDA 13.0 and Boltz for Blackwell GPUs.",
    )
    install_parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Skip PyTorch installation (use existing)",
    )
    install_parser.add_argument(
        "--skip-boltz",
        action="store_true",
        help="Skip Boltz installation",
    )
    install_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force installation even if GPU check fails",
    )
    install_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output",
    )


def _add_check_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add the check subcommand."""
    check_parser = subparsers.add_parser(
        "check",
        help="Check GPU compatibility and installation status",
        description="Verify that your GPU and software are compatible with Boltz.",
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )


def _add_predict_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add the predict subcommand."""
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run Boltz prediction with Blackwell-optimized settings",
        description="Run Boltz structure prediction with --no_kernels flag automatically set.",
    )
    predict_parser.add_argument(
        "input",
        help="Input YAML file describing the protein/complex",
    )
    predict_parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Output directory (default: output)",
    )
    predict_parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=1,
        help="Number of diffusion samples (default: 1)",
    )
    predict_parser.add_argument(
        "--recycling",
        "-r",
        type=int,
        default=3,
        help="Number of recycling steps (default: 3)",
    )
    predict_parser.add_argument(
        "--steps",
        "-t",
        type=int,
        default=200,
        help="Number of sampling steps (default: 200)",
    )
    predict_parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: 1 sample, 1 recycling, 10 steps",
    )
    predict_parser.add_argument(
        "--high-quality",
        action="store_true",
        help="High quality: 5 samples, 3 recycling, 200 steps",
    )
    predict_parser.add_argument(
        "--use-msa-server",
        action="store_true",
        help="Use ColabFold MSA server (requires internet)",
    )


def _add_manifest_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add the manifest subcommand."""
    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Generate manifest documenting hardware, software, and validation",
        description="Generate a JSON manifest with hardware info, software versions, and validation results.",
    )
    manifest_parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the prediction validation test (faster)",
    )
    manifest_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )


def dispatch_command(args: argparse.Namespace) -> int:
    """
    Dispatch to the appropriate command handler.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code from command handler
    """
    match args.command:
        case "install":
            from boltz_blackwell.installer import main as install_main

            return install_main(args)
        case "check":
            from boltz_blackwell.check import main as check_main

            return check_main(args)
        case "predict":
            from boltz_blackwell.predict import main as predict_main

            return predict_main(args)
        case "manifest":
            from boltz_blackwell.manifest import main as manifest_main

            return manifest_main(args)
        case _:
            return 0


def main() -> int:
    """
    Main entry point for boltz-blackwell command.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return dispatch_command(args)


if __name__ == "__main__":
    sys.exit(main())
