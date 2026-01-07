"""
Neural Network Library

A high-performance, from-scratch implementation of neural networks with extensive optimizations.
This library provides both standard and optimized components for building and training neural networks.

Key Features:
- Numba JIT compilation for 10-100x speedups
- Parallel processing support
- Optimized SVM implementations
- Memory-efficient operations
- Comprehensive benchmarking tools
"""

__version__ = "2.0.0"
__author__ = "Neural Network Library Team"

# Import main components for easy access
from .layers import Layer, Dense
from .activations import ReLU, Sigmoid, Tanh, Softmax
from .losses import MSELoss
from .optimizer import SGD
from .network import Sequential
from .gradient_checker import GradientChecker, check_gradients
from .autoencoder import (
    Encoder, Decoder, Autoencoder,
    create_encoder, create_decoder, create_autoencoder,
    train_autoencoder_on_mnist
)
from .data_utils import (
    MNISTDataLoader, BatchIterator, VisualizationUtils,
    create_autoencoder_data, prepare_svm_data,
    show_mnist_samples, show_reconstruction_comparison, plot_loss_curve
)

# Import optimized components
from .optimized_layers import (
    OptimizedDense, OptimizedReLU, OptimizedSigmoid, OptimizedTanh, 
    OptimizedSoftmax, BatchNormalization
)
from .optimized_losses import (
    OptimizedMSELoss, OptimizedBCEWithLogitsLoss, OptimizedCrossEntropyLoss
)
from .optimized_svm import OptimizedSVM, OptimizedSVC
from .optimized_network import OptimizedSequential, OptimizedAutoencoder
from .optimization_suite import (
    OptimizationSuite, enable_all_optimizations, create_fast_network,
    create_fast_svm, benchmark_all_components, get_performance_report
)

# Import practical optimizations (more reliable performance improvements)
from .practical_layers import (
    PracticalDense, PracticalReLU, PracticalSigmoid, PracticalTanh,
    create_practical_dense, create_practical_activation
)
from .practical_losses import (
    PracticalMSELoss, PracticalBCEWithLogitsLoss, PracticalCrossEntropyLoss,
    create_practical_loss
)

# Import SVM implementations
from .svm import SVM, SVC
from .fast_svm import FastSVM

__all__ = [
    # Original components
    'Layer', 'Dense',
    'ReLU', 'Sigmoid', 'Tanh', 'Softmax',
    'MSELoss',
    'SGD',
    'Sequential',
    'GradientChecker', 'check_gradients',
    'Encoder', 'Decoder', 'Autoencoder',
    'create_encoder', 'create_decoder', 'create_autoencoder',
    'train_autoencoder_on_mnist',
    'MNISTDataLoader', 'BatchIterator', 'VisualizationUtils',
    'create_autoencoder_data', 'prepare_svm_data',
    'show_mnist_samples', 'show_reconstruction_comparison', 'plot_loss_curve',
    
    # SVM implementations
    'SVM', 'SVC', 'FastSVM',
    
    # Optimized components
    'OptimizedDense', 'OptimizedReLU', 'OptimizedSigmoid', 'OptimizedTanh',
    'OptimizedSoftmax', 'BatchNormalization',
    'OptimizedMSELoss', 'OptimizedBCEWithLogitsLoss', 'OptimizedCrossEntropyLoss',
    'OptimizedSVM', 'OptimizedSVC',
    'OptimizedSequential', 'OptimizedAutoencoder',
    
    # Practical optimizations (recommended for most use cases)
    'PracticalDense', 'PracticalReLU', 'PracticalSigmoid', 'PracticalTanh',
    'create_practical_dense', 'create_practical_activation',
    'PracticalMSELoss', 'PracticalBCEWithLogitsLoss', 'PracticalCrossEntropyLoss',
    'create_practical_loss',
    
    # Optimization suite
    'OptimizationSuite', 'enable_all_optimizations', 'create_fast_network',
    'create_fast_svm', 'benchmark_all_components', 'get_performance_report'
]

# Auto-enable optimizations on import (can be disabled by setting environment variable)
import os
if os.environ.get('DISABLE_AUTO_OPTIMIZATION', '').lower() not in ('1', 'true', 'yes'):
    try:
        enable_all_optimizations(verbose=False)
    except Exception:
        pass  # Silently fail if optimizations can't be enabled