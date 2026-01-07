#!/usr/bin/env python3
"""
Setup script for Neural Network Library optimizations.

This script helps users install and configure the optimization dependencies
and provides system-specific recommendations for maximum performance.
"""

import subprocess
import sys
import platform
import os
import warnings

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required for optimizations")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_package(package_name, import_name=None):
    """Install a package using pip."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"📦 Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False

def check_numba_installation():
    """Check and install Numba with proper configuration."""
    print("\n" + "="*50)
    print("NUMBA JIT COMPILATION SETUP")
    print("="*50)
    
    if not install_package("numba"):
        return False
    
    try:
        import numba
        print(f"✅ Numba version: {numba.__version__}")
        
        # Test Numba compilation
        from numba import jit
        
        @jit(nopython=True)
        def test_function(x):
            return x * 2
        
        # Compile the function
        result = test_function(5.0)
        if result == 10.0:
            print("✅ Numba JIT compilation test passed")
        else:
            print("❌ Numba JIT compilation test failed")
            return False
            
    except Exception as e:
        print(f"❌ Numba test failed: {e}")
        return False
    
    return True

def check_parallel_processing():
    """Check and install parallel processing dependencies."""
    print("\n" + "="*50)
    print("PARALLEL PROCESSING SETUP")
    print("="*50)
    
    if not install_package("joblib"):
        return False
    
    try:
        import joblib
        print(f"✅ Joblib version: {joblib.__version__}")
        
        # Test parallel processing
        from joblib import Parallel, delayed
        import multiprocessing
        
        def test_parallel_function(x):
            return x ** 2
        
        n_jobs = min(4, multiprocessing.cpu_count())
        results = Parallel(n_jobs=n_jobs)(
            delayed(test_parallel_function)(i) for i in range(10)
        )
        
        expected = [i**2 for i in range(10)]
        if results == expected:
            print(f"✅ Parallel processing test passed (using {n_jobs} cores)")
        else:
            print("❌ Parallel processing test failed")
            return False
            
    except Exception as e:
        print(f"❌ Parallel processing test failed: {e}")
        return False
    
    return True

def check_scientific_computing():
    """Check scientific computing dependencies."""
    print("\n" + "="*50)
    print("SCIENTIFIC COMPUTING DEPENDENCIES")
    print("="*50)
    
    packages = [
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("scikit-learn", "sklearn")
    ]
    
    all_installed = True
    for package_name, import_name in packages:
        if not install_package(package_name, import_name):
            all_installed = False
    
    if all_installed:
        # Check versions
        try:
            import numpy as np
            import matplotlib
            import sklearn
            
            print(f"✅ NumPy version: {np.__version__}")
            print(f"✅ Matplotlib version: {matplotlib.__version__}")
            print(f"✅ Scikit-learn version: {sklearn.__version__}")
        except Exception as e:
            print(f"⚠️  Version check failed: {e}")
    
    return all_installed

def detect_system_capabilities():
    """Detect system capabilities and provide recommendations."""
    print("\n" + "="*50)
    print("SYSTEM CAPABILITIES ANALYSIS")
    print("="*50)
    
    # CPU information
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"✅ CPU cores detected: {cpu_count}")
        
        if cpu_count >= 8:
            print("🚀 High-performance system detected - all optimizations recommended")
        elif cpu_count >= 4:
            print("⚡ Multi-core system detected - parallel optimizations recommended")
        else:
            print("💻 Single/dual-core system - JIT optimizations recommended")
            
    except Exception:
        print("❌ Could not detect CPU information")
    
    # Memory information
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"✅ Available RAM: {memory_gb:.1f} GB")
        
        if memory_gb >= 16:
            print("🚀 High memory system - large batch sizes recommended")
        elif memory_gb >= 8:
            print("⚡ Adequate memory - medium batch sizes recommended")
        else:
            print("💾 Limited memory - small batch sizes recommended")
            
    except ImportError:
        print("⚠️  Install psutil for memory analysis: pip install psutil")
    except Exception:
        print("❌ Could not detect memory information")
    
    # Platform information
    system = platform.system()
    machine = platform.machine()
    print(f"✅ Platform: {system} {machine}")
    
    if system == "Windows":
        print("🪟 Windows detected - ensure Visual C++ redistributables are installed")
    elif system == "Darwin":
        print("🍎 macOS detected - optimizations should work well")
    elif system == "Linux":
        print("🐧 Linux detected - optimal performance expected")

def configure_environment():
    """Configure environment variables for optimal performance."""
    print("\n" + "="*50)
    print("ENVIRONMENT CONFIGURATION")
    print("="*50)
    
    # Numba configuration
    numba_config = {
        'NUMBA_CACHE_DIR': os.path.expanduser('~/.numba_cache'),
        'NUMBA_NUM_THREADS': str(min(8, os.cpu_count() if hasattr(os, 'cpu_count') else 4)),
        'NUMBA_THREADING_LAYER': 'workqueue'
    }
    
    print("Recommended Numba environment variables:")
    for key, value in numba_config.items():
        current_value = os.environ.get(key)
        if current_value != value:
            print(f"  export {key}={value}")
        else:
            print(f"✅ {key} already set to {value}")
    
    # NumPy configuration
    numpy_config = {
        'OMP_NUM_THREADS': '1',  # Let Numba handle threading
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1'
    }
    
    print("\nRecommended NumPy environment variables:")
    for key, value in numpy_config.items():
        current_value = os.environ.get(key)
        if current_value != value:
            print(f"  export {key}={value}")
        else:
            print(f"✅ {key} already set to {value}")

def run_performance_test():
    """Run a quick performance test to verify optimizations."""
    print("\n" + "="*50)
    print("PERFORMANCE VERIFICATION TEST")
    print("="*50)
    
    try:
        # Import the library
        sys.path.insert(0, '.')
        from lib.smart_optimizations import SmartDense
        from lib.layers import Dense
        import numpy as np
        import time
        
        # Test parameters
        batch_size = 100
        input_size = 784
        output_size = 256
        iterations = 100
        
        print(f"Running performance test...")
        print(f"  Batch size: {batch_size}")
        print(f"  Layer size: {input_size} -> {output_size}")
        print(f"  Iterations: {iterations}")
        
        # Generate test data
        inputs = np.random.randn(batch_size, input_size).astype(np.float32)
        
        # Create layers
        original_layer = Dense(input_size, output_size)
        smart_layer = SmartDense(input_size, output_size)
        
        # Set same weights for fair comparison
        smart_layer.weights = original_layer.weights.astype(np.float32)
        smart_layer.biases = original_layer.biases.astype(np.float32)
        
        # Warm-up
        for _ in range(5):
            original_layer.forward(inputs)
            smart_layer.forward(inputs)
        
        # Time original
        start_time = time.time()
        for _ in range(iterations):
            output = original_layer.forward(inputs)
            grad_output = np.random.randn(*output.shape).astype(np.float32)
            original_layer.backward(grad_output)
        original_time = time.time() - start_time
        
        # Time smart
        start_time = time.time()
        for _ in range(iterations):
            output = smart_layer.forward(inputs)
            grad_output = np.random.randn(*output.shape).astype(np.float32)
            smart_layer.backward(grad_output)
        smart_time = time.time() - start_time
        
        speedup = original_time / smart_time
        
        print(f"\nPerformance Test Results:")
        print(f"  Original time:  {original_time:.4f} seconds")
        print(f"  Optimized time: {smart_time:.4f} seconds")
        print(f"  Speedup:        {speedup:.2f}x")
        
        if speedup > 2.0:
            print("🚀 Excellent optimization performance!")
        elif speedup > 1.5:
            print("⚡ Very good optimization performance!")
        elif speedup > 1.1:
            print("✅ Good optimization performance!")
        elif speedup > 0.9:
            print("✅ Comparable performance (optimizations provide other benefits)")
        else:
            print("⚠️  Performance regression detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_optimization_guide():
    """Create a user guide for optimizations."""
    guide_content = """
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
"""
    
    with open('OPTIMIZATION_GUIDE.md', 'w') as f:
        f.write(guide_content)
    
    print("📖 Optimization guide created: OPTIMIZATION_GUIDE.md")

def main():
    """Main setup function."""
    print("Neural Network Library Optimization Setup")
    print("=" * 50)
    print("This script will help you install and configure optimizations")
    print("for maximum performance.\n")
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install and check dependencies
    if not check_scientific_computing():
        success = False
    
    if not check_numba_installation():
        success = False
    
    if not check_parallel_processing():
        success = False
    
    # System analysis
    detect_system_capabilities()
    
    # Environment configuration
    configure_environment()
    
    # Performance test
    if success:
        run_performance_test()
    
    # Create guide
    create_optimization_guide()
    
    # Final summary
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    
    if success:
        print("🎉 Optimization setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the optimization demo: python optimization_demo.py")
        print("2. Read the optimization guide: OPTIMIZATION_GUIDE.md")
        print("3. Start using optimized components in your code")
    else:
        print("⚠️  Some optimizations could not be set up.")
        print("The library will still work but with reduced performance.")
        print("Please check the error messages above and try again.")
    
    print("\nFor support, please check the documentation or open an issue.")

if __name__ == "__main__":
    main()