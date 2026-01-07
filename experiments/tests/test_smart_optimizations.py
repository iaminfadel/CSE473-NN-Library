#!/usr/bin/env python3
"""
Test smart optimizations that provide real performance improvements.
"""

import numpy as np
import time
import sys
import os

# Add lib to path
sys.path.insert(0, '.')

def test_smart_dense():
    """Test smart dense layer."""
    print("Testing Smart Dense Layer...")
    
    from lib.layers import Dense
    from lib.smart_optimizations import SmartDense
    
    # Test parameters
    batch_size = 100
    input_size = 784
    output_size = 256
    iterations = 100
    
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
    
    # Test correctness
    output_orig = original_layer.forward(inputs)
    output_smart = smart_layer.forward(inputs)
    
    max_diff = np.max(np.abs(output_orig - output_smart))
    print(f"  Max output difference: {max_diff:.6f}")
    
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
    
    print(f"  Original time:  {original_time:.4f}s")
    print(f"  Smart time:     {smart_time:.4f}s")
    print(f"  Speedup:        {speedup:.2f}x")
    
    return speedup

def test_smart_activations():
    """Test smart activation functions."""
    print("\nTesting Smart Activations...")
    
    from lib.activations import ReLU, Sigmoid
    from lib.smart_optimizations import SmartReLU, SmartSigmoid
    
    # Test different sizes to see adaptive behavior
    test_sizes = [
        (100, 50),    # Small - should use NumPy
        (1000, 512),  # Large - should use JIT
    ]
    
    activations = [
        ('ReLU', ReLU, SmartReLU),
        ('Sigmoid', Sigmoid, SmartSigmoid)
    ]
    
    results = {}
    
    for size_name, (batch_size, features) in zip(['Small', 'Large'], test_sizes):
        print(f"\n  Testing {size_name} arrays ({batch_size}x{features}):")
        
        inputs = np.random.randn(batch_size, features).astype(np.float32)
        
        for act_name, original_class, smart_class in activations:
            original = original_class()
            smart = smart_class()
            
            # Warm-up
            for _ in range(5):
                original.forward(inputs)
                smart.forward(inputs)
            
            # Time original
            start_time = time.time()
            for _ in range(200):
                output = original.forward(inputs)
                grad_output = np.random.randn(*output.shape).astype(np.float32)
                original.backward(grad_output)
            original_time = time.time() - start_time
            
            # Time smart
            start_time = time.time()
            for _ in range(200):
                output = smart.forward(inputs)
                grad_output = np.random.randn(*output.shape).astype(np.float32)
                smart.backward(grad_output)
            smart_time = time.time() - start_time
            
            speedup = original_time / smart_time
            results[f"{act_name}_{size_name}"] = speedup
            
            print(f"    {act_name}: {speedup:.2f}x speedup")
    
    return results

def test_smart_loss():
    """Test smart loss function."""
    print("\nTesting Smart Loss Function...")
    
    from lib.losses import MSELoss
    from lib.smart_optimizations import SmartMSELoss
    
    # Test parameters
    batch_size = 1000
    output_size = 100
    iterations = 500
    
    print(f"  Batch size: {batch_size}")
    print(f"  Output size: {output_size}")
    print(f"  Iterations: {iterations}")
    
    # Generate test data
    predictions = np.random.randn(batch_size, output_size).astype(np.float32)
    targets = np.random.randn(batch_size, output_size).astype(np.float32)
    
    # Create loss functions
    original_loss = MSELoss()
    smart_loss = SmartMSELoss()
    
    # Test correctness
    loss_orig = original_loss.forward(predictions, targets)
    loss_smart = smart_loss.forward(predictions, targets)
    
    loss_diff = abs(loss_orig - loss_smart)
    print(f"  Loss difference: {loss_diff:.6f}")
    
    # Warm-up
    for _ in range(10):
        original_loss.forward(predictions, targets)
        smart_loss.forward(predictions, targets)
    
    # Time original
    start_time = time.time()
    for _ in range(iterations):
        loss = original_loss.forward(predictions, targets)
        grad = original_loss.backward(predictions, targets)
    original_time = time.time() - start_time
    
    # Time smart
    start_time = time.time()
    for _ in range(iterations):
        loss = smart_loss.forward(predictions, targets)
        grad = smart_loss.backward(predictions, targets)
    smart_time = time.time() - start_time
    
    speedup = original_time / smart_time
    
    print(f"  Original time:  {original_time:.4f}s")
    print(f"  Smart time:     {smart_time:.4f}s")
    print(f"  Speedup:        {speedup:.2f}x")
    
    return speedup

def test_size_scaling():
    """Test how optimizations scale with size."""
    print("\nTesting Size Scaling...")
    
    from lib.smart_optimizations import SmartReLU, should_use_jit
    
    sizes = [100, 1000, 10000, 100000]
    
    for size in sizes:
        use_jit = should_use_jit(size)
        print(f"  Size {size:6d}: {'JIT' if use_jit else 'NumPy'}")

def main():
    """Run smart optimization tests."""
    print("Smart Optimization Test Suite")
    print("=" * 50)
    
    try:
        # Test individual components
        dense_speedup = test_smart_dense()
        activation_speedups = test_smart_activations()
        loss_speedup = test_smart_loss()
        
        # Test scaling behavior
        test_size_scaling()
        
        # Summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        
        print(f"Dense Layer Speedup:     {dense_speedup:.2f}x")
        print(f"MSE Loss Speedup:        {loss_speedup:.2f}x")
        
        for key, speedup in activation_speedups.items():
            print(f"{key} Speedup: {speedup:.2f}x")
        
        # Calculate overall performance
        all_speedups = [dense_speedup, loss_speedup] + list(activation_speedups.values())
        geometric_mean = np.prod(all_speedups) ** (1.0 / len(all_speedups))
        
        print(f"\nGeometric Mean Speedup: {geometric_mean:.2f}x")
        
        if geometric_mean > 1.2:
            print("🚀 Smart optimizations are working excellently!")
        elif geometric_mean > 1.0:
            print("⚡ Smart optimizations provide good improvements")
        elif geometric_mean > 0.9:
            print("✅ Smart optimizations maintain performance with other benefits")
        else:
            print("⚠️  Smart optimizations need investigation")
        
        print("\nKey Benefits of Smart Optimizations:")
        print("• Adaptive algorithm selection based on data size")
        print("• Float32 precision for better cache performance")
        print("• Contiguous memory layout optimization")
        print("• Leverages NumPy's optimized BLAS when appropriate")
        print("• Uses JIT compilation only when beneficial")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()