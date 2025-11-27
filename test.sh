#!/bin/bash
# Test script for Boltz on Blackwell GPUs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=============================================="
echo "Boltz Test Suite for Blackwell GPUs"
echo "=============================================="

TESTS_PASSED=0
TESTS_FAILED=0

# Test function
run_test() {
    local name="$1"
    local cmd="$2"
    echo -e "\n${YELLOW}Testing: $name${NC}"
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}  ✗ FAILED${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Test 1: Python import
run_test "Boltz import" "python3 -c 'import boltz'"

# Test 2: PyTorch CUDA
run_test "PyTorch CUDA" "python3 -c 'import torch; assert torch.cuda.is_available()'"

# Test 3: CUDA 13.0
run_test "CUDA 13.0" "python3 -c 'import torch; assert \"13\" in torch.version.cuda'"

# Test 4: sm_120 support
run_test "sm_120 architecture" "python3 -c 'import torch; assert \"sm_120\" in torch.cuda.get_arch_list()'"

# Test 5: Boltz CLI
run_test "Boltz CLI" "boltz --help"

# Test 6: Boltz predict help
run_test "Boltz predict" "boltz predict --help"

# Test 7: GPU tensor allocation
run_test "GPU tensor allocation" "python3 -c 'import torch; x = torch.rand(100, 100).cuda(); print(x.sum())'"

# Test 8: Dependencies
echo -e "\n${YELLOW}Testing: Dependencies${NC}"
if python3 -c "
import sys
deps = [
    ('torch', 'PyTorch'),
    ('pytorch_lightning', 'PyTorch Lightning'),
    ('rdkit', 'RDKit'),
    ('numpy', 'NumPy'),
]
all_ok = True
for module, name in deps:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ✓ {name}: {version}')
    except ImportError:
        print(f'  ✗ {name}: NOT FOUND')
        all_ok = False
sys.exit(0 if all_ok else 1)
"; then
    echo -e "${GREEN}  ✓ PASSED${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}  ✗ FAILED${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Test 9: Full prediction test
echo -e "\n${YELLOW}Testing: Full prediction (insulin chain A)${NC}"
echo "  This may take a minute on first run (model download)..."

rm -rf "$SCRIPT_DIR/test_output"
mkdir -p "$SCRIPT_DIR/test_output"

if boltz predict "$SCRIPT_DIR/examples/insulin_a.yaml" \
    --out_dir "$SCRIPT_DIR/test_output" \
    --diffusion_samples 1 \
    --recycling_steps 1 \
    --sampling_steps 10 \
    --no_kernels 2>&1 | tail -5; then
    
    # Check for output file
    if ls "$SCRIPT_DIR/test_output"/boltz_results_*/predictions/*/confidence_*.json 1> /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ PASSED - Structure predicted successfully${NC}"
        
        # Show confidence score
        CONF_FILE=$(ls "$SCRIPT_DIR/test_output"/boltz_results_*/predictions/*/confidence_*.json 2>/dev/null | head -1)
        if [ -f "$CONF_FILE" ]; then
            echo -e "  Confidence scores:"
            python3 -c "
import json
with open('$CONF_FILE') as f:
    data = json.load(f)
print(f\"    confidence_score: {data.get('confidence_score', 'N/A'):.3f}\")
print(f\"    plddt: {data.get('complex_plddt', 'N/A'):.3f}\")
"
        fi
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}  ✗ FAILED - No output file generated${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
else
    echo -e "${RED}  ✗ FAILED - Prediction error${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi

# Summary
echo -e "\n=============================================="
echo "Test Summary"
echo "=============================================="
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed! ✓${NC}"
    echo -e "\nBoltz is ready to use. Example:"
    echo "  boltz predict examples/insulin_a.yaml --out_dir output/ --no_kernels"
    exit 0
else
    echo -e "\n${RED}Some tests failed. Check the output above.${NC}"
    exit 1
fi
