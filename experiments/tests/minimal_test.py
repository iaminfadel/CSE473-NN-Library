#!/usr/bin/env python3
"""
Minimal test to check basic functionality.
"""

import numpy as np
import time

def test_numpy_performance():
    """Test basic NumPy performance."""
    print("Testing NumPy Performance...")
    
    # Create test data
    size = 10000
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    
    # Test ReLU
    start_time = time.time()
    for _ in range(1000):
        result = np.maximum(0, a)
    relu_time = time.time() - start_time
    
    # Test matrix multiplication
    A = np.random.randn(100, 100).astype(np.float32)
    B = np.random.randn(100, 100).astype(np.float32)
    
    start_time = time.time()
    for _ in range(100):
        C = np.dot(A, B)
    matmul_time = time.time() - start_time
    
    print(f"ReLU time (1000 iterations): {relu_time:.4f}s")
    print(f"MatMul time (100 iterations): {matmul_time:.4f}s")
    
    return relu_time, matmul_time

def test_numba_basic():
    """Test basic Numba functionality."""
    print("\nTesting Numba...")
    
    try:
        from numba import jit
        
        @jit(nopython=True)
        def simple_function(x):
            return x * 2
        
        # Test compilation
        result = simple_function(5.0)
        if result == 10.0:
            print("✅ Numba basic test passed")
            return True
        else:
            print("❌ Numba basic test failed")
            return False
            
    except Exception as e:
        print(f"❌ Numba import/compilation failed: {e}")
        return False

def main():
    print("Minimal Performance Test")
    print("=" * 30)
    
    # Test NumPy
    relu_time, matmul_time = test_numpy_performance()
    
    # Test Numba
    numba_works = test_numba_basic()
    
    print("\nSummary:")
    print(f"NumPy ReLU: {relu_time:.4f}s")
    print(f"NumPy MatMul: {matmul_time:.4f}s")
    print(f"Numba works: {numba_works}")

if __name__ == "__main__":
    main()