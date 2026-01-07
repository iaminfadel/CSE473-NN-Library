"""
Comprehensive optimization suite for the neural network library.

This module provides a unified interface to all optimizations, benchmarking tools,
and performance analysis utilities. It serves as the main entry point for users
who want to leverage the full optimization capabilities of the library.
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
from joblib import Parallel, delayed

# Import all optimized components
from .optimized_ops import set_numba_threads, get_optimal_num_threads
from .optimized_layers import (
    OptimizedDense, OptimizedReLU, OptimizedSigmoid, OptimizedTanh, 
    OptimizedSoftmax, BatchNormalization
)
from .optimized_losses import (
    OptimizedMSELoss, OptimizedBCEWithLogitsLoss, OptimizedCrossEntropyLoss
)
from .optimized_svm import OptimizedSVM
from .optimized_network import OptimizedSequential, OptimizedAutoencoder

# Import original components for comparison
from .layers import Dense
from .activations import ReLU, Sigmoid, Tanh, Softmax
from .losses import MSELoss
from .svm import SVM
from .network import Sequential


class OptimizationSuite:
    """
    Comprehensive optimization suite providing unified access to all performance enhancements.
    
    This class serves as the main interface for users who want to leverage optimized
    implementations throughout their neural network workflows.
    """
    
    def __init__(self, enable_all_optimizations=True, n_jobs=-1, verbose=True):
        """
        Initialize the optimization suite.
        
        Args:
            enable_all_optimizations: Whether to enable all available optimizations
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Whether to print optimization status
        """
        self.enable_all_optimizations = enable_all_optimizations
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Initialize optimization status
        self.optimizations_enabled = {
            'numba_jit': False,
            'parallel_processing': False,
            'optimized_layers': False,
            'optimized_losses': False,
            'optimized_svm': False,
            'batch_normalization': False
        }
        
        if enable_all_optimizations:
            self.enable_optimizations()
    
    def enable_optimizations(self):
        """Enable all available optimizations."""
        try:
            # Enable Numba JIT compilation
            set_numba_threads()
            self.optimizations_enabled['numba_jit'] = True
            
            # Enable parallel processing
            if self.n_jobs != 1:
                self.optimizations_enabled['parallel_processing'] = True
            
            # Enable optimized components
            self.optimizations_enabled['optimized_layers'] = True
            self.optimizations_enabled['optimized_losses'] = True
            self.optimizations_enabled['optimized_svm'] = True
            self.optimizations_enabled['batch_normalization'] = True
            
            if self.verbose:
                self._print_optimization_status()
                
        except Exception as e:
            warnings.warn(f"Failed to enable some optimizations: {e}")
    
    def _print_optimization_status(self):
        """Print the current optimization status."""
        print("Neural Network Library Optimization Suite")
        print("=" * 50)
        
        for opt_name, enabled in self.optimizations_enabled.items():
            status = "✅ ENABLED" if enabled else "❌ DISABLED"
            print(f"{opt_name.replace('_', ' ').title()}: {status}")
        
        if self.optimizations_enabled['numba_jit']:
            print(f"Numba threads: {get_optimal_num_threads()}")
        
        if self.optimizations_enabled['parallel_processing']:
            print(f"Parallel jobs: {self.n_jobs}")
        
        print("=" * 50)
    
    def create_optimized_network(self, architecture='mlp', **kwargs):
        """
        Create an optimized neural network based on architecture type.
        
        Args:
            architecture: Type of network ('mlp', 'autoencoder', 'classifier')
            **kwargs: Architecture-specific parameters
            
        Returns:
            Optimized network instance
        """
        use_fast_ops = self.optimizations_enabled['optimized_layers']
        
        if architecture == 'mlp':
            layer_sizes = kwargs.get('layer_sizes', [784, 128, 64, 10])
            activations = kwargs.get('activations', None)
            
            network = OptimizedSequential(use_fast_ops=use_fast_ops, n_jobs=self.n_jobs)
            
            if activations is None:
                activations = ['relu'] * (len(layer_sizes) - 2) + ['softmax']
            
            for i in range(len(layer_sizes) - 1):
                network.add(OptimizedDense(layer_sizes[i], layer_sizes[i + 1], use_fast_ops))
                
                if i < len(activations):
                    activation = activations[i].lower()
                    if activation == 'relu':
                        network.add(OptimizedReLU(use_fast_ops))
                    elif activation == 'sigmoid':
                        network.add(OptimizedSigmoid(use_fast_ops))
                    elif activation == 'tanh':
                        network.add(OptimizedTanh(use_fast_ops))
                    elif activation == 'softmax':
                        network.add(OptimizedSoftmax(use_fast_ops))
            
            return network
        
        elif architecture == 'autoencoder':
            input_dim = kwargs.get('input_dim', 784)
            latent_dim = kwargs.get('latent_dim', 32)
            hidden_dims = kwargs.get('hidden_dims', None)
            
            return OptimizedAutoencoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                use_fast_ops=use_fast_ops
            )
        
        elif architecture == 'classifier':
            input_size = kwargs.get('input_size', 784)
            num_classes = kwargs.get('num_classes', 10)
            hidden_sizes = kwargs.get('hidden_sizes', [128, 64])
            
            layer_sizes = [input_size] + hidden_sizes + [num_classes]
            activations = ['relu'] * len(hidden_sizes) + ['softmax']
            
            return self.create_optimized_network(
                architecture='mlp',
                layer_sizes=layer_sizes,
                activations=activations
            )
        
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def create_optimized_svm(self, **kwargs):
        """
        Create an optimized SVM classifier.
        
        Args:
            **kwargs: SVM parameters
            
        Returns:
            OptimizedSVM instance
        """
        if not self.optimizations_enabled['optimized_svm']:
            warnings.warn("Optimized SVM not enabled, falling back to standard SVM")
            return SVM(**kwargs)
        
        # Set default n_jobs if not specified
        if 'n_jobs' not in kwargs:
            kwargs['n_jobs'] = self.n_jobs
        
        return OptimizedSVM(**kwargs)
    
    def create_optimized_loss(self, loss_type='mse', **kwargs):
        """
        Create an optimized loss function.
        
        Args:
            loss_type: Type of loss function
            **kwargs: Loss function parameters
            
        Returns:
            Optimized loss function instance
        """
        use_fast_ops = self.optimizations_enabled['optimized_losses']
        
        loss_map = {
            'mse': OptimizedMSELoss,
            'bce_logits': OptimizedBCEWithLogitsLoss,
            'cross_entropy': OptimizedCrossEntropyLoss
        }
        
        if loss_type not in loss_map:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return loss_map[loss_type](use_fast_ops=use_fast_ops, **kwargs)
    
    def benchmark_performance(self, components=None, input_shapes=None, iterations=100):
        """
        Comprehensive performance benchmarking of optimized vs original components.
        
        Args:
            components: List of components to benchmark (None for all)
            input_shapes: Dictionary of input shapes for different components
            iterations: Number of iterations for timing
            
        Returns:
            Dictionary with benchmark results
        """
        if components is None:
            components = ['dense', 'relu', 'sigmoid', 'mse_loss', 'svm']
        
        if input_shapes is None:
            input_shapes = {
                'dense': (100, 784),
                'activation': (100, 128),
                'loss': (100, 10),
                'svm': (1000, 20)
            }
        
        results = {}
        
        if self.verbose:
            print("Running comprehensive performance benchmarks...")
            print("=" * 60)
        
        # Benchmark Dense layers
        if 'dense' in components:
            results['dense'] = self._benchmark_dense_layers(
                input_shapes['dense'], iterations
            )
        
        # Benchmark activation functions
        if any(act in components for act in ['relu', 'sigmoid', 'tanh']):
            results['activations'] = self._benchmark_activations(
                input_shapes['activation'], iterations, components
            )
        
        # Benchmark loss functions
        if any(loss in components for loss in ['mse_loss', 'bce_loss']):
            results['losses'] = self._benchmark_losses(
                input_shapes['loss'], iterations, components
            )
        
        # Benchmark SVM
        if 'svm' in components:
            results['svm'] = self._benchmark_svm(
                input_shapes['svm'], iterations
            )
        
        if self.verbose:
            self._print_benchmark_results(results)
        
        return results
    
    def _benchmark_dense_layers(self, input_shape, iterations):
        """Benchmark Dense layer implementations."""
        # Original Dense layer
        original_layer = Dense(input_shape[1], input_shape[1])
        
        # Optimized Dense layer
        optimized_layer = OptimizedDense(
            input_shape[1], input_shape[1], 
            use_fast_ops=self.optimizations_enabled['optimized_layers']
        )
        
        # Generate test data
        inputs = np.random.randn(*input_shape).astype(np.float32)
        
        # Benchmark original
        original_times = self._time_layer_operations(original_layer, inputs, iterations)
        
        # Benchmark optimized
        optimized_times = self._time_layer_operations(optimized_layer, inputs, iterations)
        
        speedup = original_times['total'] / optimized_times['total']
        
        return {
            'original': original_times,
            'optimized': optimized_times,
            'speedup': speedup,
            'component': 'Dense Layer'
        }
    
    def _benchmark_activations(self, input_shape, iterations, components):
        """Benchmark activation function implementations."""
        results = {}
        
        activation_pairs = [
            ('relu', ReLU, OptimizedReLU),
            ('sigmoid', Sigmoid, OptimizedSigmoid),
            ('tanh', Tanh, OptimizedTanh)
        ]
        
        inputs = np.random.randn(*input_shape).astype(np.float32)
        
        for name, original_class, optimized_class in activation_pairs:
            if name in components:
                original_layer = original_class()
                optimized_layer = optimized_class(
                    use_fast_ops=self.optimizations_enabled['optimized_layers']
                )
                
                original_times = self._time_layer_operations(original_layer, inputs, iterations)
                optimized_times = self._time_layer_operations(optimized_layer, inputs, iterations)
                
                speedup = original_times['total'] / optimized_times['total']
                
                results[name] = {
                    'original': original_times,
                    'optimized': optimized_times,
                    'speedup': speedup,
                    'component': f'{name.upper()} Activation'
                }
        
        return results
    
    def _benchmark_losses(self, input_shape, iterations, components):
        """Benchmark loss function implementations."""
        results = {}
        
        predictions = np.random.randn(*input_shape).astype(np.float32)
        targets = np.random.randn(*input_shape).astype(np.float32)
        
        if 'mse_loss' in components:
            original_loss = MSELoss()
            optimized_loss = OptimizedMSELoss(
                use_fast_ops=self.optimizations_enabled['optimized_losses']
            )
            
            original_times = self._time_loss_operations(
                original_loss, predictions, targets, iterations
            )
            optimized_times = self._time_loss_operations(
                optimized_loss, predictions, targets, iterations
            )
            
            speedup = original_times['total'] / optimized_times['total']
            
            results['mse'] = {
                'original': original_times,
                'optimized': optimized_times,
                'speedup': speedup,
                'component': 'MSE Loss'
            }
        
        return results
    
    def _benchmark_svm(self, input_shape, iterations):
        """Benchmark SVM implementations."""
        # Generate synthetic binary classification data
        X = np.random.randn(*input_shape).astype(np.float64)
        y = (np.random.randn(input_shape[0]) > 0).astype(int)
        
        # Original SVM
        original_svm = SVM(max_iter=50, tol=1e-2)
        
        # Optimized SVM
        optimized_svm = OptimizedSVM(
            max_iter=50, tol=1e-2, n_jobs=self.n_jobs
        )
        
        # Time original SVM
        start_time = time.time()
        original_svm.fit(X, y)
        original_fit_time = time.time() - start_time
        
        start_time = time.time()
        original_svm.predict(X)
        original_predict_time = time.time() - start_time
        
        # Time optimized SVM
        start_time = time.time()
        optimized_svm.fit(X, y)
        optimized_fit_time = time.time() - start_time
        
        start_time = time.time()
        optimized_svm.predict(X)
        optimized_predict_time = time.time() - start_time
        
        original_total = original_fit_time + original_predict_time
        optimized_total = optimized_fit_time + optimized_predict_time
        speedup = original_total / optimized_total
        
        return {
            'original': {
                'fit_time': original_fit_time,
                'predict_time': original_predict_time,
                'total': original_total
            },
            'optimized': {
                'fit_time': optimized_fit_time,
                'predict_time': optimized_predict_time,
                'total': optimized_total
            },
            'speedup': speedup,
            'component': 'SVM Classifier'
        }
    
    def _time_layer_operations(self, layer, inputs, iterations):
        """Time forward and backward operations for a layer."""
        # Warm-up
        for _ in range(10):
            output = layer.forward(inputs)
            if hasattr(layer, 'backward'):
                grad_output = np.random.randn(*output.shape).astype(np.float32)
                layer.backward(grad_output)
        
        # Time forward pass
        start_time = time.time()
        for _ in range(iterations):
            output = layer.forward(inputs)
        forward_time = (time.time() - start_time) / iterations
        
        # Time backward pass
        grad_output = np.random.randn(*output.shape).astype(np.float32)
        start_time = time.time()
        for _ in range(iterations):
            if hasattr(layer, 'backward'):
                layer.backward(grad_output)
        backward_time = (time.time() - start_time) / iterations
        
        return {
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total': forward_time + backward_time
        }
    
    def _time_loss_operations(self, loss_fn, predictions, targets, iterations):
        """Time forward and backward operations for a loss function."""
        # Warm-up
        for _ in range(10):
            loss_value = loss_fn.forward(predictions, targets)
            gradient = loss_fn.backward(predictions, targets)
        
        # Time forward pass
        start_time = time.time()
        for _ in range(iterations):
            loss_value = loss_fn.forward(predictions, targets)
        forward_time = (time.time() - start_time) / iterations
        
        # Time backward pass
        start_time = time.time()
        for _ in range(iterations):
            gradient = loss_fn.backward(predictions, targets)
        backward_time = (time.time() - start_time) / iterations
        
        return {
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total': forward_time + backward_time
        }
    
    def _print_benchmark_results(self, results):
        """Print formatted benchmark results."""
        print("\nBenchmark Results Summary")
        print("=" * 60)
        
        for category, category_results in results.items():
            if isinstance(category_results, dict) and 'speedup' in category_results:
                # Single component result
                self._print_single_benchmark(category_results)
            else:
                # Multiple components in category
                print(f"\n{category.upper()} BENCHMARKS:")
                print("-" * 40)
                for component, result in category_results.items():
                    self._print_single_benchmark(result)
    
    def _print_single_benchmark(self, result):
        """Print a single benchmark result."""
        component = result['component']
        speedup = result['speedup']
        
        original_time = result['original']['total']
        optimized_time = result['optimized']['total']
        
        print(f"{component}:")
        print(f"  Original:  {original_time*1000:.3f} ms")
        print(f"  Optimized: {optimized_time*1000:.3f} ms")
        print(f"  Speedup:   {speedup:.2f}x")
        print()
    
    def analyze_memory_usage(self, network, input_shape, batch_sizes=None):
        """
        Analyze memory usage patterns for different batch sizes.
        
        Args:
            network: Network to analyze
            input_shape: Shape of input data (without batch dimension)
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with memory usage analysis
        """
        if batch_sizes is None:
            batch_sizes = [1, 16, 32, 64, 128, 256]
        
        memory_usage = {}
        
        for batch_size in batch_sizes:
            full_input_shape = (batch_size,) + input_shape
            inputs = np.random.randn(*full_input_shape).astype(np.float32)
            
            # Measure memory before and after forward pass
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            output = network.forward(inputs)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            memory_usage[batch_size] = {
                'memory_used_mb': memory_used,
                'memory_per_sample_kb': (memory_used * 1024) / batch_size,
                'output_shape': output.shape
            }
        
        return memory_usage
    
    def get_optimization_recommendations(self, use_case='general'):
        """
        Get optimization recommendations based on use case.
        
        Args:
            use_case: Type of use case ('general', 'training', 'inference', 'large_data')
            
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            'general': {
                'enable_jit': True,
                'use_parallel': True,
                'batch_size': 32,
                'dtype': 'float32',
                'description': 'Balanced optimizations for general use'
            },
            'training': {
                'enable_jit': True,
                'use_parallel': True,
                'batch_size': 64,
                'dtype': 'float32',
                'use_batch_norm': True,
                'description': 'Optimizations focused on training speed'
            },
            'inference': {
                'enable_jit': True,
                'use_parallel': False,
                'batch_size': 1,
                'dtype': 'float32',
                'description': 'Optimizations for fast inference'
            },
            'large_data': {
                'enable_jit': True,
                'use_parallel': True,
                'batch_size': 128,
                'dtype': 'float32',
                'memory_efficient': True,
                'description': 'Optimizations for large datasets'
            }
        }
        
        if use_case not in recommendations:
            use_case = 'general'
        
        return recommendations[use_case]


# Global optimization suite instance
optimization_suite = OptimizationSuite()


# Convenience functions for easy access
def enable_all_optimizations(verbose=True):
    """Enable all available optimizations globally."""
    global optimization_suite
    optimization_suite.enable_all_optimizations = True
    optimization_suite.verbose = verbose
    optimization_suite.enable_optimizations()


def create_fast_network(architecture='mlp', **kwargs):
    """Create an optimized network with all optimizations enabled."""
    return optimization_suite.create_optimized_network(architecture, **kwargs)


def create_fast_svm(**kwargs):
    """Create an optimized SVM with all optimizations enabled."""
    return optimization_suite.create_optimized_svm(**kwargs)


def benchmark_all_components(verbose=True):
    """Run comprehensive benchmarks on all components."""
    return optimization_suite.benchmark_performance()


def get_performance_report():
    """Get a comprehensive performance report."""
    results = benchmark_all_components(verbose=False)
    
    report = "Neural Network Library Performance Report\n"
    report += "=" * 50 + "\n\n"
    
    total_speedup = 1.0
    component_count = 0
    
    for category, category_results in results.items():
        if isinstance(category_results, dict) and 'speedup' in category_results:
            speedup = category_results['speedup']
            total_speedup *= speedup
            component_count += 1
            report += f"{category_results['component']}: {speedup:.2f}x speedup\n"
        else:
            for component, result in category_results.items():
                speedup = result['speedup']
                total_speedup *= speedup
                component_count += 1
                report += f"{result['component']}: {speedup:.2f}x speedup\n"
    
    geometric_mean_speedup = total_speedup ** (1.0 / component_count)
    report += f"\nGeometric Mean Speedup: {geometric_mean_speedup:.2f}x\n"
    
    return report