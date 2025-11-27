#!/bin/bash
# Convenience wrapper for boltz predict with Blackwell GPU flags

set -e

# Default values
DIFFUSION_SAMPLES=1
RECYCLING_STEPS=3
SAMPLING_STEPS=200
OUTPUT_DIR="output"

# Help
show_help() {
    echo "Usage: $0 <input.yaml> [options]"
    echo ""
    echo "Wrapper for boltz predict with Blackwell GPU optimizations."
    echo ""
    echo "Options:"
    echo "  -o, --output DIR       Output directory (default: output)"
    echo "  -s, --samples N        Number of diffusion samples (default: 1)"
    echo "  -r, --recycling N      Number of recycling steps (default: 3)"
    echo "  -t, --steps N          Number of sampling steps (default: 200)"
    echo "  --fast                 Fast mode: 1 sample, 1 recycling, 10 steps"
    echo "  --high-quality         High quality: 5 samples, 3 recycling, 200 steps"
    echo "  --use-msa-server       Use ColabFold MSA server (requires internet)"
    echo "  -h, --help             Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 examples/insulin_a.yaml"
    echo "  $0 examples/insulin_a.yaml --fast"
    echo "  $0 examples/insulin_a.yaml --high-quality -o results/"
}

# Parse arguments
INPUT_FILE=""
USE_MSA_SERVER=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--samples)
            DIFFUSION_SAMPLES="$2"
            shift 2
            ;;
        -r|--recycling)
            RECYCLING_STEPS="$2"
            shift 2
            ;;
        -t|--steps)
            SAMPLING_STEPS="$2"
            shift 2
            ;;
        --fast)
            DIFFUSION_SAMPLES=1
            RECYCLING_STEPS=1
            SAMPLING_STEPS=10
            shift
            ;;
        --high-quality)
            DIFFUSION_SAMPLES=5
            RECYCLING_STEPS=3
            SAMPLING_STEPS=200
            shift
            ;;
        --use-msa-server)
            USE_MSA_SERVER="--use_msa_server"
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$INPUT_FILE" ]; then
                INPUT_FILE="$1"
            else
                EXTRA_ARGS="$EXTRA_ARGS $1"
            fi
            shift
            ;;
    esac
done

# Check input file
if [ -z "$INPUT_FILE" ]; then
    echo "Error: No input file specified"
    show_help
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Run prediction
echo "Running Boltz prediction..."
echo "  Input: $INPUT_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  Samples: $DIFFUSION_SAMPLES"
echo "  Recycling: $RECYCLING_STEPS"
echo "  Steps: $SAMPLING_STEPS"
echo ""

boltz predict "$INPUT_FILE" \
    --out_dir "$OUTPUT_DIR" \
    --diffusion_samples "$DIFFUSION_SAMPLES" \
    --recycling_steps "$RECYCLING_STEPS" \
    --sampling_steps "$SAMPLING_STEPS" \
    --no_kernels \
    $USE_MSA_SERVER \
    $EXTRA_ARGS

echo ""
echo "Done! Results saved to: $OUTPUT_DIR"
