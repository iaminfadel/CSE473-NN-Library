
# Neural Network Library Optimization Guide

## Quick Start
```python
from lib import enable_all_optimizations, create_fast_network, create_fast_svm

# Enable all optimizations
enable_all_optimizations()

# Create optimized neural network
network = create_fast_network('mlp', layer_sizes=[784, 128, 64, 10])

# Create optimized SVM
svm = create_fast_svm(kernel='rbf', C=1.0)
```

## Performance Tips

### 1. Use Optimized Components
- Replace `Dense` with `OptimizedDense`
- Replace `ReLU` with `OptimizedReLU`
- Replace `MSELoss` with `OptimizedMSELoss`
- Replace `SVM` with `OptimizedSVM`

### 2. Batch Size Optimization
- Larger batch sizes generally perform better with optimizations
- Recommended batch sizes: 32-128 for training, 1-16 for inference
- Monitor memory usage with large batch sizes

### 3. Data Type Optimization
- Use `np.float32` instead of `np.float64` when possible
- Ensure input data is contiguous in memory

### 4. Parallel Processing
- Set `n_jobs=-1` for maximum parallelization
- For small datasets, `n_jobs=1` might be faster due to overhead

### 5. Environment Variables
```bash
export NUMBA_NUM_THREADS=8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Benchmarking
```python
from lib import benchmark_all_components, get_performance_report

# Run comprehensive benchmarks
results = benchmark_all_components()

# Get performance report
print(get_performance_report())
```

## Troubleshooting

### Numba Issues
- Ensure you have the latest Numba version
- Clear Numba cache: `rm -rf ~/.numba_cache`
- Check for LLVM compatibility issues

### Memory Issues
- Reduce batch size
- Use gradient checkpointing for large networks
- Monitor memory usage with `psutil`

### Performance Issues
- Verify optimizations are enabled
- Check CPU utilization
- Profile code to identify bottlenecks
