# CUDA Kernel Optimization

## Quick Start

```bash
make clean && make
./baseline.x
```

## Performance Results
- **GPU**: NVIDIA H100 80GB HBM3
- **Build flags**: `-O3 -lineinfo -Xcompiler -fopenmp`
- **Original kernel**: ~14.54ms
- **Optimized kernel**: ~1.48ms
- **Speedup**: 9.8x improvement

## Key Optimizations
- Grid-stride loops ensure all elements processed on any GPU
- Fast math intrinsics (`__logf`, `__fsqrt_rn`, etc.) for better performance
- Reduced warp divergence by moving pattern branches outside iteration loops
- Maintains numerical accuracy within 0.1% tolerance