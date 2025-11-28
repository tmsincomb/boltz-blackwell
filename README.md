# Boltz for Blackwell GPUs

Installation and testing scripts for running [Boltz](https://github.com/jwohlwend/boltz) biomolecular structure prediction on NVIDIA Blackwell architecture GPUs (compute capability 12.1).

## Supported Hardware

- **NVIDIA GB10** (DGX Spark)
- **NVIDIA GB100** (B100)
- **NVIDIA GB200** (B200)
- Other Blackwell architecture GPUs with compute capability 12.1

## The Problem

Standard PyTorch releases don't support Blackwell GPUs (sm_121) yet. This causes:
- `nvrtc: error: invalid value for --gpu-architecture (-arch)` during runtime kernel compilation
- cuEquivariance kernel failures

## The Solution

Use **PyTorch nightly with CUDA 13.0** which includes:
- `sm_120` architecture support with forward compatibility to sm_121
- Updated nvrtc that handles Blackwell GPUs

Run Boltz with `--no_kernels` flag to disable cuEquivariance kernels.

## Quick Start (PyPI)

The easiest way to get started:

```bash
# Install boltz-blackwell from PyPI
pip install boltz-blackwell

# Install PyTorch nightly + Boltz with one command
boltz-blackwell install

# Verify everything is working
boltz-blackwell check

# Run a prediction (--no_kernels is automatic)
boltz-blackwell predict input.yaml --output results/
```

## Installation from Source

```bash
# Clone this repo
git clone https://github.com/tmsincomb/boltz-blackwell.git
cd boltz-blackwell

# Install the package
pip install -e .

# Install PyTorch and Boltz
boltz-blackwell install

# Or use the shell script
./install.sh
```

## Manual Installation

If you prefer to install manually:

```bash
# 1. Install PyTorch nightly with CUDA 13.0
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# 2. Install Boltz
pip install boltz[cuda] -U

# 3. Verify installation
python -c "import torch; print('CUDA:', torch.version.cuda); print('Arch:', torch.cuda.get_arch_list())"
```

## Usage

### Using boltz-blackwell CLI (Recommended)

The `boltz-blackwell` CLI automatically adds the `--no_kernels` flag:

```bash
# Basic prediction
boltz-blackwell predict input.yaml

# Fast mode (quick test)
boltz-blackwell predict input.yaml --fast

# High quality mode
boltz-blackwell predict input.yaml --high-quality

# Custom settings
boltz-blackwell predict input.yaml --samples 3 --steps 100 --output results/

# With MSA server (best accuracy, requires internet)
boltz-blackwell predict input.yaml --use-msa-server
```

### Using boltz directly

```bash
# Single-sequence mode (fully local, fast but lower accuracy)
boltz predict input.yaml --out_dir output/ --no_kernels
```

### High-Quality Prediction

```bash
# More samples and steps (slower, better quality)
boltz predict input.yaml --out_dir output/ --no_kernels \
    --diffusion_samples 5 \
    --recycling_steps 3 \
    --sampling_steps 200
```

### With MSA (Best Accuracy)

For best accuracy, provide a pre-computed MSA or use the ColabFold server:

```bash
# With MSA server (requires internet)
boltz predict input.yaml --out_dir output/ --no_kernels --use_msa_server
```

## Input Format

Create a YAML file describing your protein/complex:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: GIVEQCCTSICSLYQLENYCN
      msa: empty  # Use 'empty' for single-sequence mode
```

See `examples/` for more input file examples.

## Key Flags for Blackwell GPUs

| Flag | Required | Description |
|------|----------|-------------|
| `--no_kernels` | **Yes** | Disables cuEquivariance kernels (required for sm_121) |
| `--diffusion_samples N` | No | Number of structure samples (default: 1) |
| `--recycling_steps N` | No | Number of recycling iterations (default: 3) |
| `--sampling_steps N` | No | Number of diffusion steps (default: 200) |

## Benchmarks

Tested on NVIDIA GB10 (DGX Spark):

| Configuration | Speed | Notes |
|---------------|-------|-------|
| CUDA 12.8 | ❌ | JIT compilation fails |
| CUDA 12.9 | 0.40 it/s | Works |
| **CUDA 13.0** | **0.67 it/s** | **Recommended** |

## Troubleshooting

### "nvrtc: error: invalid value for --gpu-architecture"

Your PyTorch doesn't support Blackwell GPUs. Install the CUDA 13.0 nightly:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

### "no kernel image is available for execution on the device"

Same issue - need PyTorch with sm_120/sm_121 support.

### "CUDA error: no kernel image is available"

Make sure you're using `--no_kernels` flag with boltz.

### cuEquivariance errors

The cuEquivariance library doesn't support sm_121 yet. Always use `--no_kernels`.

## Requirements

- Python 3.10+
- NVIDIA Blackwell GPU (compute capability 12.1)
- CUDA 13.0+ drivers
- ~16GB GPU memory for small proteins
- ~40GB+ GPU memory for larger complexes

## License

This installation guide is MIT licensed. Boltz itself is under [MIT License](https://github.com/jwohlwend/boltz/blob/main/LICENSE).

## References

- [Boltz GitHub](https://github.com/jwohlwend/boltz)
- [Boltz-1 Paper](https://doi.org/10.1101/2024.11.19.624167)
- [Boltz-2 Paper](https://doi.org/10.1101/2025.06.14.659707)
- [PyTorch Nightly Builds](https://pytorch.org/get-started/locally/)
