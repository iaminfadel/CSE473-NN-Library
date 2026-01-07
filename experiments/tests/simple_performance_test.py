#!/usr/bin/env python3
"""
Simple performance test to identify optimization issues.
"""

import numpy as np
import time
import sys
import os

# Disable Numba warnings
import warnings
warnings.filterwarnings('ignore')

def test_basic_operations():
    """Test basic NumPy operations vs manual implementations."""
    print("Testing Basic Operations...")
    
    # Test data
    size = 100000
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    iterations = 1000
    
    print(f"Array size: {size}, Iterations: {iterations}")
    
    # Test 1: ReLU operation
    print("\n1. ReLU Operation:")
    
    # NumPy version
    start_time = time.time()
    for _ in range(iterations):
        result = np.maximum(0, a)
    numpy_time = time.time() - start_time
    
    # Manual loop version
    start_time = time.time()
    for _ in range(iterations):
        result = np.zeros_like(a)
        for i in range(len(a)):
            result[i] = max(0, a[i])
    manual_time = time.time() - start_time
    
    print(f"  NumPy time:  {numpy_time:.4f}s")
    print(f"  Manual time: {manual_time:.4f}s")
    print(f"  NumPy speedup: {manual_time / numpy_time:.2f}x")
    
    # Test 2: Matrix multiplication
    print("\n2. Matrix Multiplication:")
    
    # Create matrices
    m, n, k = 100, 100, 100
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(k, n).astype(np.float32)
    
    # NumPy version
    start_time = time.time()
    for _ in range(100):
        C = np.dot(A, B)
    numpy_time = time.time() - start_time
    
    # Manual version
    start_time = time.time()
    for _ in range(100):
        C = np.zeros((m, n), dtype=np.float32)
        for i in range(m):
            for j in range(n):
                for l in range(k):
                    C[i, j] += A[i, l] * B[l, j]
    manual_time = time.time() - start_time
    
    print(f"  NumPy time:  {numpy_time:.4f}s")
    print(f"  Manual time: {manual_time:.4f}s")
    print(f"  NumPy speedup: {manual_time / numpy_time:.2f}x")

def test_numba_compilation():
    """Test if Numba compilation is working correctly."""
    print("\n" + "="*50)
    print("Testing Numba Compilation...")
    
    try:
        from numba import jit
        
        @jit(nopython=True, cache=True)
        def numba_relu(x):
            result = np.zeros_like(x)
            for i in range(len(x)):
                result[i] = max(0.0, x[i])
            return result
        
        # Test data
        size = 10000
        x = np.random.randn(size).astype(np.float32)
        iterations = 100
        
        print(f"Array size: {size}, Iterations: {iterations}")
        
        # First call (includes compilation time)
        print("First call (with compilation)...")
        start_time = time.time()
        result_numba = numba_relu(x)
        first_call_time = time.time() - start_time
        print(f"  First call time: {first_call_time:.4f}s")
        
        # Subsequent calls (compiled)
        print("Subsequent calls (compiled)...")
        start_time = time.time()
        for _ in range(iterations):
            result_numba = numba_relu(x)
        numba_time = time.time() - start_time
        
        # NumPy version
        start_time = time.time()
        for _ in range(iterations):
            result_numpy = np.maximum(0, x)
        numpy_time = time.time() - start_time
        
        print(f"  Numba time:  {numba_time:.4f}s")
        print(f"  NumPy time:  {numpy_time:.4f}s")
        print(f"  Speedup:     {numpy_time / numba_time:.2f}x")
        
        # Check correctness
        max_diff = np.max(np.abs(result_numba - result_numpy))
        print(f"  Max difference: {max_diff:.6f}")
        
        if max_diff < 1e-6:
            print("  ✅ Correctness test passed")
        else:
            print("  ❌ Correctness test failed")
        
        return numba_time < numpy_time
        
    except Exception as e:
        print(f"❌ Numba test failed: {e}")
        return False

def test_layer_operations():
    """Test actual layer operations."""
    print("\n" + "="*50)
    print("Testing Layer Operations...")
    
    try:
        sys.path.insert(0, '.')
        from lib.layers import Dense
        
        # Test parameters
        batch_size = 32  # Smaller batch size
        input_size = 100  # Smaller layer size
        output_size = 50
        iterations = 50   # Fewer iterations
        
        print(f"Batch size: {batch_size}")
        print(f"Layer size: {input_size} -> {output_size}")
        print(f"Iterations: {iterations}")
        
        # Generate test data
        inputs = np.random.randn(batch_size, input_size).astype(np.float32)
        
        # Create layer
        layer = Dense(input_size, output_size)
        
        # Warm-up
        for _ in range(5):
            output = layer.forward(inputs)
            grad_output = np.random.randn(*output.shape).astype(np.float32)
            layer.backward(grad_output)
        
        # Time the operations
        start_time = time.time()
        for _ in range(iterations):
            output = layer.forward(inputs)
            grad_output = np.random.randn(*output.shape).astype(np.float32)
            layer.backward(grad_output)
        total_time = time.time() - start_time
        
        print(f"Total time: {total_time:.4f}s")
        print(f"Time per iteration: {total_time/iterations*1000:.2f}ms")
        
        return total_time
        
    except Exception as e:
        print(f"❌ Layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run simple performance tests."""
    print("Simple Performance Test Suite")
    print("=" * 50)
    
    # Test basic operations
    test_basic_operations()
    
    # Test Numba compilation
    numba_works = test_numba_compilation()
    
    # Test layer operations
    layer_time = test_layer_operations()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if numba_works:
        print("✅ Numba compilation is working")
    else:
        print("❌ Numba compilation has issues")
    
    if layer_time is not None:
        print(f"✅ Layer operations completed in {layer_time:.4f}s")
    else:
        print("❌ Layer operations failed")
    
    print("\nRecommendations:")
    print("1. NumPy's optimized BLAS is very fast for matrix operations")
    print("2. Numba is best for element-wise operations on large arrays")
    print("3. For small arrays, NumPy overhead is often lower than JIT compilation")
    print("4. Focus on data type optimization (float32) and memory layout")

if __name__ == "__main__":
    main()