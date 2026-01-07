#!/usr/bin/env python3
"""
Test script for practical optimizations.

This script tests the practical optimization implementations to ensure
they provide real performance improvements without sacrificing correctness.
"""

import numpy as np
import time
import sys
import os

# Add the lib directory to the path
sys.path.insert(0, '.')

def test_practical_dense():
    """Test practical dense layer optimization."""
    print("Testing Practical Dense Layer...")
    
    from lib.layers import Dense
    from lib.practical_layers import PracticalDense
    
    # Test parameters
    batch_size = 100
    input_size = 784
    output_size = 256
    iterations = 100
    
    # Generate test data
    inputs = np.random.randn(batch_size, input_size).astype(np.float32)
    
    # Create layers
    original_layer = Dense(input_size, output_size)
    practical_layer = PracticalDense(input_size, output_size)
    
    # Set same weights for fair comparison
    practical_layer.weights = original_layer.weights.astype(np.float32)
    practical_layer.biases = original_layer.biases.astype(np.float32)
    
    # Test correctness first
    output_orig = original_layer.forward(inputs)
    output_prac = practical_layer.forward(inputs)
    
    # Check if outputs are close (allowing for float32 vs float64 differences)
    max_diff = np.max(np.abs(output_orig - output_prac))
    print(f"  Max output difference: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print("  ✅ Correctness test passed")
    else:
        print("  ⚠️  Large output difference detected")
    
    # Performance test
    print(f"  Running {iterations} iterations...")
    
    # Warm-up
    for _ in range(5):
        original_layer.forward(inputs)
        practical_layer.forward(inputs)
    
    # Time original
    start_time = time.time()
    for _ in range(iterations):
        output = original_layer.forward(inputs)
        grad_output = np.random.randn(*output.shape).astype(np.float32)
        original_layer.backward(grad_output)
    original_time = time.time() - start_time
    
    # Time practical
    start_time = time.time()
    for _ in range(iterations):
        output = practical_layer.forward(inputs)
        grad_output = np.random.randn(*output.shape).astype(np.float32)
        practical_layer.backward(grad_output)
    practical_time = time.time() - start_time
    
    speedup = original_time / practical_time
    
    print(f"  Original time:  {original_time:.4f}s")
    print(f"  Practical time: {practical_time:.4f}s")
    print(f"  Speedup:        {speedup:.2f}x")
    
    return speedup

def test_practical_activations():
    """Test practical activation functions."""
    print("\nTesting Practical Activation Functions...")
    
    from lib.activations import ReLU, Sigmoid
    from lib.practical_layers import PracticalReLU, PracticalSigmoid
    
    # Test parameters
    batch_size = 1000
    features = 512
    iterations = 200
    
    # Generate test data
    inputs = np.random.randn(batch_size, features).astype(np.float32)
    
    activations = [
        ('ReLU', ReLU(), PracticalReLU()),
        ('Sigmoid', Sigmoid(), PracticalSigmoid())
    ]
    
    results = {}
    
    for name, original, practical in activations:
        print(f"\n  Testing {name}...")
        
        # Test correctness
        output_orig = original.forward(inputs)
        output_prac = practical.forward(inputs)
        
        max_diff = np.max(np.abs(output_orig - output_prac))
        print(f"    Max output difference: {max_diff:.6f}")
        
        # Performance test
        # Warm-up
        for _ in range(5):
            original.forward(inputs)
            practical.forward(inputs)
        
        # Time original
        start_time = time.time()
        for _ in range(iterations):
            output = original.forward(inputs)
            grad_output = np.random.randn(*output.shape).astype(np.float32)
            original.backward(grad_output)
        original_time = time.time() - start_time
        
        # Time practical
        start_time = time.time()
        for _ in range(iterations):
            output = practical.forward(inputs)
            grad_output = np.random.randn(*output.shape).astype(np.float32)
            practical.backward(grad_output)
        practical_time = time.time() - start_time
        
        speedup = original_time / practical_time
        results[name] = speedup
        
        print(f"    Original time:  {original_time:.4f}s")
        print(f"    Practical time: {practical_time:.4f}s")
        print(f"    Speedup:        {speedup:.2f}x")
    
    return results

def test_practical_losses():
    """Test practical loss functions."""
    print("\nTesting Practical Loss Functions...")
    
    from lib.losses import MSELoss
    from lib.practical_losses import PracticalMSELoss
    
    # Test parameters
    batch_size = 1000
    output_size = 10
    iterations = 500
    
    # Generate test data
    predictions = np.random.randn(batch_size, output_size).astype(np.float32)
    targets = np.random.randn(batch_size, output_size).astype(np.float32)
    
    # Create loss functions
    original_loss = MSELoss()
    practical_loss = PracticalMSELoss()
    
    # Test correctness
    loss_orig = original_loss.forward(predictions, targets)
    loss_prac = practical_loss.forward(predictions, targets)
    
    loss_diff = abs(loss_orig - loss_prac)
    print(f"  Loss difference: {loss_diff:.6f}")
    
    # Performance test
    # Warm-up
    for _ in range(10):
        original_loss.forward(predictions, targets)
        practical_loss.forward(predictions, targets)
    
    # Time original
    start_time = time.time()
    for _ in range(iterations):
        loss_value = original_loss.forward(predictions, targets)
        gradient = original_loss.backward(predictions, targets)
    original_time = time.time() - start_time
    
    # Time practical
    start_time = time.time()
    for _ in range(iterations):
        loss_value = practical_loss.forward(predictions, targets)
        gradient = practical_loss.backward(predictions, targets)
    practical_time = time.time() - start_time
    
    speedup = original_time / practical_time
    
    print(f"  Original time:  {original_time:.4f}s")
    print(f"  Practical time: {practical_time:.4f}s")
    print(f"  Speedup:        {speedup:.2f}x")
    
    return speedup

def test_array_size_scaling():
    """Test how optimizations scale with array size."""
    print("\nTesting Array Size Scaling...")
    
    from lib.practical_layers import PracticalReLU
    from lib.activations import ReLU
    
    sizes = [100, 1000, 10000, 100000]
    
    for size in sizes:
        print(f"\n  Testing size {size}...")
        
        inputs = np.random.randn(size).astype(np.float32)
        
        original = ReLU()
        practical = PracticalReLU()
        
        # Time original
        start_time = time.time()
        for _ in range(100):
            output = original.forward(inputs)
        original_time = time.time() - start_time
        
        # Time practical
        start_time = time.time()
        for _ in range(100):
            output = practical.forward(inputs)
        practical_time = time.time() - start_time
        
        speedup = original_time / practical_time
        print(f"    Speedup: {speedup:.2f}x")

def main():
    """Run all practical optimization tests."""
    print("Practical Optimization Test Suite")
    print("=" * 50)
    
    try:
        # Test individual components
        dense_speedup = test_practical_dense()
        activation_speedups = test_practical_activations()
        loss_speedup = test_practical_losses()
        
        # Test scaling behavior
        test_array_size_scaling()
        
        # Summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        
        print(f"Dense Layer Speedup:     {dense_speedup:.2f}x")
        print(f"ReLU Speedup:           {activation_speedups.get('ReLU', 0):.2f}x")
        print(f"Sigmoid Speedup:        {activation_speedups.get('Sigmoid', 0):.2f}x")
        print(f"MSE Loss Speedup:       {loss_speedup:.2f}x")
        
        # Calculate geometric mean speedup
        speedups = [dense_speedup, loss_speedup] + list(activation_speedups.values())
        geometric_mean = np.prod(speedups) ** (1.0 / len(speedups))
        
        print(f"\nGeometric Mean Speedup: {geometric_mean:.2f}x")
        
        if geometric_mean > 1.2:
            print("🚀 Practical optimizations are working well!")
        elif geometric_mean > 1.0:
            print("⚡ Practical optimizations provide modest improvements")
        else:
            print("⚠️  Practical optimizations need investigation")
        
        print("\nNote: Speedups may vary based on system configuration and array sizes.")
        print("The practical optimizations focus on reliability over maximum speed.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()