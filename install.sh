#!/bin/bash
# Install script for Boltz on Blackwell GPUs (sm_121 / compute capability 12.1)
# Tested on: NVIDIA GB10 (DGX Spark), GB100, GB200

set -e

echo "=============================================="
echo "Boltz Installation for Blackwell GPUs"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for NVIDIA GPU
echo -e "\n${YELLOW}Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${GREEN}Found GPU: $GPU_NAME (Compute Capability: $COMPUTE_CAP)${NC}"
    
    if [[ "$COMPUTE_CAP" != "12.1" ]]; then
        echo -e "${YELLOW}Warning: This script is optimized for Blackwell GPUs (compute capability 12.1)${NC}"
        echo -e "${YELLOW}Your GPU has compute capability $COMPUTE_CAP${NC}"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo -e "${RED}nvidia-smi not found. Please install NVIDIA drivers.${NC}"
    echo ""
    echo "To install NVIDIA drivers for Blackwell GPUs:"
    echo ""
    echo "  # 1. Add NVIDIA's official repository"
    echo "  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring.deb"
    echo "  sudo dpkg -i cuda-keyring.deb"
    echo "  sudo apt update"
    echo ""
    echo "  # 2. Install the driver (580+ required for Blackwell)"
    echo "  sudo apt install nvidia-driver-580"
    echo ""
    echo "  # 3. Reboot"
    echo "  sudo reboot"
    echo ""
    echo "For other distros, see: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check Python
echo -e "\n${YELLOW}Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}$PYTHON_VERSION${NC}"
else
    echo -e "${RED}Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Install PyTorch nightly with CUDA 13.0
echo -e "\n${YELLOW}Installing PyTorch nightly with CUDA 13.0...${NC}"
echo "This provides sm_120 support with forward compatibility for sm_121 (Blackwell)"

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130 --upgrade

# Verify PyTorch installation
echo -e "\n${YELLOW}Verifying PyTorch installation...${NC}"
python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Arch List: {torch.cuda.get_arch_list()}')
"

# Install Boltz
echo -e "\n${YELLOW}Installing Boltz...${NC}"
pip install boltz[cuda] -U

# Verify Boltz installation
echo -e "\n${YELLOW}Verifying Boltz installation...${NC}"
python3 -c "
import boltz
print(f'Boltz Version: {getattr(boltz, \"__version__\", \"unknown\")}')
"

# Check boltz CLI
if command -v boltz &> /dev/null; then
    echo -e "${GREEN}Boltz CLI is available${NC}"
else
    echo -e "${RED}Boltz CLI not found in PATH${NC}"
    exit 1
fi

echo -e "\n${GREEN}=============================================="
echo "Installation Complete!"
echo "=============================================="
echo -e "${NC}"
echo "To run a prediction:"
echo "  boltz predict examples/insulin_a.yaml --out_dir output/ --no_kernels"
echo ""
echo "To run tests:"
echo "  ./test.sh"
echo ""
echo -e "${YELLOW}IMPORTANT: Always use --no_kernels flag on Blackwell GPUs${NC}"
