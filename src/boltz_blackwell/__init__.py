"""
Boltz-Blackwell: Easy installation for Boltz on NVIDIA Blackwell GPUs.

This package provides utilities to install and run Boltz biomolecular
structure prediction on Blackwell architecture GPUs (compute capability 12.1).
"""

__version__ = "0.1.0"

from boltz_blackwell.installer import install
from boltz_blackwell.check import check_gpu
from boltz_blackwell.predict import predict
from boltz_blackwell.manifest import generate_manifest

__all__ = ["install", "check_gpu", "predict", "generate_manifest", "__version__"]
