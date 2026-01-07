#!/usr/bin/env python3
"""
Neural Network Library Optimization Demo

This script demonstrates the performance improvements achieved through various
optimizations including Numba JIT compilation, parallel processing, and
algorithmic improvements.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from lib import *

def generate_synthetic_data(n_samples=1000, n_features=20, n_classes=2, random_state=42):
    """Generate synthetic classification data."""
    np.random.seed(random_state)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create separable classes
    w = np.random.randn(n_features)
    y = (X @ w + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    return X, y

def demo_layer_optimizations():
    """Demonstrate layer optimization performance."""
    print("=" * 60)
    print("LAYER OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Test parameters
    batch_size = 100
    input_size = 784
    output_size = 256
    iterations = 1000
    
    print(f"Testing Dense layer: {input_size} -> {output_size}")
    print(f"Batch size: {batch_size}, Iterations: {iterations}")
    print()
    
    # Generate test data
    inputs = np.random.randn(batch_size, input_size).astype(np.float32)
    
    # Original Dense layer
    print("Testing original Dense layer...")
    original_layer = Dense(input_size, output_size)
    
    start_time = time.time()
    for _ in range(iterations):
        output = original_layer.forward(inputs)
        grad_output = np.random.randn(*output.shape).astype(np.float32)
        original_layer.backward(grad_output)
    original_time = time.time() - start_time
    
    # Optimized Dense layer
    print("Testing optimized Dense layer...")
    optimized_layer = OptimizedDense(input_size, output_size, use_fast_ops=True)
    
    start_time = time.time()
    for _ in range(iterations):
        output = optimized_layer.forward(inputs)
        grad_output = np.random.randn(*output.shape).astype(np.float32)
        optimized_layer.backward(grad_output)
    optimized_time = time.time() - start_time
    
    speedup = original_time / optimized_time
    
    print(f"Original time:  {original_time:.4f} seconds")
    print(f"Optimized time: {optimized_time:.4f} seconds")
    print(f"Speedup:        {speedup:.2f}x")
    print()

def demo_activation_optimizations():
    """Demonstrate activation function optimization performance."""
    print("=" * 60)
    print("ACTIVATION OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Test parameters
    batch_size = 1000
    features = 512
    iterations = 1000
    
    print(f"Testing activations: batch_size={batch_size}, features={features}")
    print(f"Iterations: {iterations}")
    print()
    
    # Generate test data
    inputs = np.random.randn(batch_size, features).astype(np.float32)
    
    activations = [
        ('ReLU', ReLU, OptimizedReLU),
        ('Sigmoid', Sigmoid, OptimizedSigmoid),
        ('Tanh', Tanh, OptimizedTanh)
    ]
    
    for name, original_class, optimized_class in activations:
        print(f"Testing {name} activation...")
        
        # Original activation
        original_activation = original_class()
        start_time = time.time()
        for _ in range(iterations):
            output = original_activation.forward(inputs)
            grad_output = np.random.randn(*output.shape).astype(np.float32)
            original_activation.backward(grad_output)
        original_time = time.time() - start_time
        
        # Optimized activation
        optimized_activation = optimized_class(use_fast_ops=True)
        start_time = time.time()
        for _ in range(iterations):
            output = optimized_activation.forward(inputs)
            grad_output = np.random.randn(*output.shape).astype(np.float32)
            optimized_activation.backward(grad_output)
        optimized_time = time.time() - start_time
        
        speedup = original_time / optimized_time
        
        print(f"  Original:  {original_time:.4f}s")
        print(f"  Optimized: {optimized_time:.4f}s")
        print(f"  Speedup:   {speedup:.2f}x")
        print()

def demo_loss_optimizations():
    """Demonstrate loss function optimization performance."""
    print("=" * 60)
    print("LOSS FUNCTION OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Test parameters
    batch_size = 1000
    output_size = 10
    iterations = 1000
    
    print(f"Testing loss functions: batch_size={batch_size}, output_size={output_size}")
    print(f"Iterations: {iterations}")
    print()
    
    # Generate test data
    predictions = np.random.randn(batch_size, output_size).astype(np.float32)
    targets = np.random.randn(batch_size, output_size).astype(np.float32)
    
    print("Testing MSE Loss...")
    
    # Original MSE loss
    original_loss = MSELoss()
    start_time = time.time()
    for _ in range(iterations):
        loss_value = original_loss.forward(predictions, targets)
        gradient = original_loss.backward(predictions, targets)
    original_time = time.time() - start_time
    
    # Optimized MSE loss
    optimized_loss = OptimizedMSELoss(use_fast_ops=True)
    start_time = time.time()
    for _ in range(iterations):
        loss_value = optimized_loss.forward(predictions, targets)
        gradient = optimized_loss.backward(predictions, targets)
    optimized_time = time.time() - start_time
    
    speedup = original_time / optimized_time
    
    print(f"  Original:  {original_time:.4f}s")
    print(f"  Optimized: {optimized_time:.4f}s")
    print(f"  Speedup:   {speedup:.2f}x")
    print()

def demo_svm_optimizations():
    """Demonstrate SVM optimization performance."""
    print("=" * 60)
    print("SVM OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Generate synthetic data
    n_samples = 1000
    n_features = 20
    
    print(f"Generating synthetic data: {n_samples} samples, {n_features} features")
    X, y = generate_synthetic_data(n_samples, n_features)
    
    print("Training SVMs...")
    print()
    
    # Original SVM
    print("Testing original SVM...")
    original_svm = SVM(C=1.0, kernel='linear', max_iter=100, tol=1e-3)
    
    start_time = time.time()
    original_svm.fit(X, y)
    fit_time_original = time.time() - start_time
    
    start_time = time.time()
    predictions_original = original_svm.predict(X)
    predict_time_original = time.time() - start_time
    
    accuracy_original = np.mean(predictions_original == y)
    
    # Optimized SVM
    print("Testing optimized SVM...")
    optimized_svm = OptimizedSVM(C=1.0, kernel='linear', max_iter=100, tol=1e-3, n_jobs=-1)
    
    start_time = time.time()
    optimized_svm.fit(X, y)
    fit_time_optimized = time.time() - start_time
    
    start_time = time.time()
    predictions_optimized = optimized_svm.predict(X)
    predict_time_optimized = time.time() - start_time
    
    accuracy_optimized = np.mean(predictions_optimized == y)
    
    # Calculate speedups
    fit_speedup = fit_time_original / fit_time_optimized
    predict_speedup = predict_time_original / predict_time_optimized
    total_speedup = (fit_time_original + predict_time_original) / (fit_time_optimized + predict_time_optimized)
    
    print(f"Original SVM:")
    print(f"  Fit time:     {fit_time_original:.4f}s")
    print(f"  Predict time: {predict_time_original:.4f}s")
    print(f"  Accuracy:     {accuracy_original:.4f}")
    print()
    
    print(f"Optimized SVM:")
    print(f"  Fit time:     {fit_time_optimized:.4f}s")
    print(f"  Predict time: {predict_time_optimized:.4f}s")
    print(f"  Accuracy:     {accuracy_optimized:.4f}")
    print()
    
    print(f"Speedups:")
    print(f"  Fit speedup:     {fit_speedup:.2f}x")
    print(f"  Predict speedup: {predict_speedup:.2f}x")
    print(f"  Total speedup:   {total_speedup:.2f}x")
    print()

def demo_network_training():
    """Demonstrate optimized network training performance."""
    print("=" * 60)
    print("NETWORK TRAINING OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Generate synthetic data for autoencoder
    n_samples = 2000
    input_dim = 784
    
    print(f"Generating synthetic autoencoder data: {n_samples} samples, {input_dim} features")
    X = np.random.rand(n_samples, input_dim).astype(np.float32)
    
    # Split into train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # Training parameters
    epochs = 20
    batch_size = 32
    latent_dim = 32
    learning_rate = 0.01
    
    print(f"Training parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Learning rate: {learning_rate}")
    print()
    
    # Original autoencoder
    print("Training original autoencoder...")
    original_autoencoder = create_autoencoder(latent_dim=latent_dim, input_dim=input_dim)
    
    start_time = time.time()
    original_history = original_autoencoder.train(
        X_train, X_test, epochs=epochs, batch_size=batch_size, 
        learning_rate=learning_rate, print_interval=epochs//2
    )
    original_training_time = time.time() - start_time
    
    # Optimized autoencoder
    print("\nTraining optimized autoencoder...")
    optimized_autoencoder = OptimizedAutoencoder(
        input_dim=input_dim, latent_dim=latent_dim, use_fast_ops=True
    )
    
    start_time = time.time()
    optimized_history = optimized_autoencoder.train(
        X_train, X_test, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, verbose=True
    )
    optimized_training_time = time.time() - start_time
    
    # Calculate speedup
    training_speedup = original_training_time / optimized_training_time
    
    print(f"\nTraining Results:")
    print(f"Original training time:  {original_training_time:.2f}s")
    print(f"Optimized training time: {optimized_training_time:.2f}s")
    print(f"Training speedup:        {training_speedup:.2f}x")
    
    # Compare final losses
    original_final_loss = original_history['train_losses'][-1]
    optimized_final_loss = optimized_history['losses'][-1]
    
    print(f"Original final loss:     {original_final_loss:.6f}")
    print(f"Optimized final loss:    {optimized_final_loss:.6f}")
    print()

def demo_comprehensive_benchmark():
    """Run comprehensive benchmarks using the optimization suite."""
    print("=" * 60)
    print("COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 60)
    
    # Create optimization suite
    suite = OptimizationSuite(enable_all_optimizations=True, verbose=True)
    
    print("\nRunning comprehensive benchmarks...")
    results = suite.benchmark_performance(
        components=['dense', 'relu', 'sigmoid', 'mse_loss', 'svm'],
        iterations=500
    )
    
    print("\nBenchmark Summary:")
    print("-" * 40)
    
    total_speedup = 1.0
    component_count = 0
    
    for category, category_results in results.items():
        if isinstance(category_results, dict) and 'speedup' in category_results:
            speedup = category_results['speedup']
            component = category_results['component']
            total_speedup *= speedup
            component_count += 1
            print(f"{component}: {speedup:.2f}x speedup")
        else:
            for component_name, result in category_results.items():
                speedup = result['speedup']
                component = result['component']
                total_speedup *= speedup
                component_count += 1
                print(f"{component}: {speedup:.2f}x speedup")
    
    geometric_mean_speedup = total_speedup ** (1.0 / component_count)
    print(f"\nGeometric Mean Speedup: {geometric_mean_speedup:.2f}x")
    print()

def plot_performance_comparison():
    """Create performance comparison plots."""
    print("=" * 60)
    print("PERFORMANCE VISUALIZATION")
    print("=" * 60)
    
    # Test different batch sizes for Dense layer performance
    batch_sizes = [16, 32, 64, 128, 256, 512]
    input_size = 784
    output_size = 256
    iterations = 100
    
    original_times = []
    optimized_times = []
    speedups = []
    
    print("Testing performance across different batch sizes...")
    
    for batch_size in batch_sizes:
        print(f"  Testing batch size {batch_size}...")
        
        inputs = np.random.randn(batch_size, input_size).astype(np.float32)
        
        # Original layer
        original_layer = Dense(input_size, output_size)
        start_time = time.time()
        for _ in range(iterations):
            output = original_layer.forward(inputs)
            grad_output = np.random.randn(*output.shape).astype(np.float32)
            original_layer.backward(grad_output)
        original_time = time.time() - start_time
        
        # Optimized layer
        optimized_layer = OptimizedDense(input_size, output_size, use_fast_ops=True)
        start_time = time.time()
        for _ in range(iterations):
            output = optimized_layer.forward(inputs)
            grad_output = np.random.randn(*output.shape).astype(np.float32)
            optimized_layer.backward(grad_output)
        optimized_time = time.time() - start_time
        
        original_times.append(original_time)
        optimized_times.append(optimized_time)
        speedups.append(original_time / optimized_time)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Performance comparison plot
    ax1.plot(batch_sizes, original_times, 'o-', label='Original', linewidth=2, markersize=8)
    ax1.plot(batch_sizes, optimized_times, 's-', label='Optimized', linewidth=2, markersize=8)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Dense Layer Performance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Speedup plot
    ax2.plot(batch_sizes, speedups, 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Optimization Speedup vs Batch Size')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('optimization_performance.png', dpi=150, bbox_inches='tight')
    print(f"\nPerformance plots saved as 'optimization_performance.png'")
    plt.show()

def main():
    """Run all optimization demos."""
    print("Neural Network Library Optimization Demo")
    print("This demo showcases the performance improvements achieved through")
    print("Numba JIT compilation, parallel processing, and algorithmic optimizations.")
    print()
    
    # Enable all optimizations
    enable_all_optimizations(verbose=True)
    print()
    
    try:
        # Run individual component demos
        demo_layer_optimizations()
        demo_activation_optimizations()
        demo_loss_optimizations()
        demo_svm_optimizations()
        demo_network_training()
        
        # Run comprehensive benchmark
        demo_comprehensive_benchmark()
        
        # Create performance plots
        plot_performance_comparison()
        
        # Print final performance report
        print("=" * 60)
        print("FINAL PERFORMANCE REPORT")
        print("=" * 60)
        print(get_performance_report())
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Some optimizations may not be available on this system.")
        print("Please ensure numba and joblib are installed:")
        print("  pip install numba joblib")

if __name__ == "__main__":
    main()