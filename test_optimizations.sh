#!/bin/bash

# CUDA Kernel Optimization Testing Script
# This script compiles and tests both original and optimized versions

echo "=== CUDA Kernel Optimization Testing ==="
echo "Testing on: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'GPU info not available')"
echo

# Check if NVCC is available
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please ensure CUDA toolkit is installed and in PATH."
    exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -f original.x optimized.x

# Compile original version
echo "Compiling original version..."
nvcc -O3 -lineinfo -Xcompiler -fopenmp cuda_prog_original.cu -o original.x
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile original version"
    exit 1
fi

# Compile optimized version
echo "Compiling optimized version..."
nvcc -O3 -lineinfo -Xcompiler -fopenmp cuda_prog.cu -o optimized.x
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile optimized version"
    exit 1
fi

echo
echo "=== TESTING CORRECTNESS ==="

# Test original version
echo "Testing original version correctness..."
./original.x > original_output.txt 2>&1
ORIGINAL_EXIT=$?

# Test optimized version
echo "Testing optimized version correctness..."
./optimized.x > optimized_output.txt 2>&1
OPTIMIZED_EXIT=$?

# Check if both completed successfully
if [ $ORIGINAL_EXIT -ne 0 ]; then
    echo "ERROR: Original version failed to execute"
    cat original_output.txt
    exit 1
fi

if [ $OPTIMIZED_EXIT -ne 0 ]; then
    echo "ERROR: Optimized version failed to execute"
    cat optimized_output.txt
    exit 1
fi

# Extract results and timing
ORIGINAL_CORRECT=$(grep "Results are correct\|FAIL" original_output.txt)
OPTIMIZED_CORRECT=$(grep "Results are correct\|FAIL" optimized_output.txt)

echo "Original result: $ORIGINAL_CORRECT"
echo "Optimized result: $OPTIMIZED_CORRECT"

# Extract timing information
ORIGINAL_TIME=$(grep "A:" original_output.txt | awk '{print $2}')
OPTIMIZED_TIME=$(grep "A:" optimized_output.txt | awk '{print $2}')

echo
echo "=== PERFORMANCE COMPARISON ==="
echo "Original time:  ${ORIGINAL_TIME} ms"
echo "Optimized time: ${OPTIMIZED_TIME} ms"

# Calculate speedup
if [ ! -z "$ORIGINAL_TIME" ] && [ ! -z "$OPTIMIZED_TIME" ]; then
    SPEEDUP=$(python3 -c "print(f'{float('$ORIGINAL_TIME') / float('$OPTIMIZED_TIME'):.2f}')" 2>/dev/null || echo "calculation failed")
    echo "Speedup: ${SPEEDUP}x"
    echo

    # Performance assessment
    if (( $(echo "$SPEEDUP > 5" | bc -l 2>/dev/null || echo 0) )); then
        echo "✅ EXCELLENT: Achieved >5x speedup!"
    elif (( $(echo "$SPEEDUP > 2" | bc -l 2>/dev/null || echo 0) )); then
        echo "✅ GOOD: Achieved >2x speedup"
    elif (( $(echo "$SPEEDUP > 1.2" | bc -l 2>/dev/null || echo 0) )); then
        echo "⚠️  MODERATE: Some improvement achieved"
    else
        echo "❌ POOR: Little to no improvement"
    fi
fi

echo
echo "=== PROFILING COMMANDS ==="
echo "For detailed analysis, run:"
echo "nvprof ./original.x"
echo "nvprof ./optimized.x"
echo
echo "For occupancy analysis:"
echo "nvprof --metrics achieved_occupancy ./original.x"
echo "nvprof --metrics achieved_occupancy ./optimized.x"

echo
echo "=== FILES GENERATED ==="
echo "- original.x: Compiled original version"
echo "- optimized.x: Compiled optimized version"
echo "- original_output.txt: Original execution output"
echo "- optimized_output.txt: Optimized execution output"