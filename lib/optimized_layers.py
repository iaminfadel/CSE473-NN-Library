"""
Optimized layer implementations using Numba JIT compilation and parallel processing.

This module provides high-performance versions of neural network layers
with significant speed improvements over the base implementations.
"""

import numpy as np
from .layers import Layer
from .optimized_ops import (
    fast_matrix_multiply, fast_add_bias, fast_gradient_weights, fast_gradient_bias,
    fast_relu_forward, fast_relu_backward,
    fast_sigmoid_forward, fast_sigmoid_backward,
    fast_tanh_forward, fast_tanh_backward
)


class OptimizedDense(Layer):
    """
    Optimized fully connected (dense) layer using Numba JIT compilation.
    
    This layer performs the same operations as Dense but with significant
    performance improvements through JIT compilation and parallel processing.
    """
    
    def __init__(self, input_size, output_size, use_fast_ops=True):
        """
        Initialize Optimized Dense layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            use_fast_ops: Whether to use JIT-compiled operations (default: True)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.use_fast_ops = use_fast_ops
        
        # Initialize weights using Xavier/Glorot initialization
        xavier_std = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size).astype(np.float32) * xavier_std
        
        # Initialize biases to zero
        self.biases = np.zeros((1, output_size), dtype=np.float32)
        
        # Storage for gradients
        self.grad_weights = None
        self.grad_biases = None
        
        # Storage for inputs (needed for backward pass)
        self.last_inputs = None
    
    def forward(self, inputs):
        """
        Optimized forward pass through dense layer.
        
        Args:
            inputs: Input data of shape (batch_size, input_size)
            
        Returns:
            Output of shape (batch_size, output_size)
        """
        # Ensure inputs are float32 for optimal performance
        inputs = inputs.astype(np.float32)
        
        # Store inputs for backward pass
        self.last_inputs = inputs.copy()
        
        # For dense layers, NumPy's BLAS-optimized operations are typically faster
        # than our JIT implementations for most practical sizes
        # The optimization comes from using float32 and ensuring contiguous arrays
        if self.use_fast_ops:
            # Ensure arrays are contiguous for better cache performance
            inputs_contig = np.ascontiguousarray(inputs)
            weights_contig = np.ascontiguousarray(self.weights)
            
            # Use optimized operations for specific cases
            if inputs.shape[0] * inputs.shape[1] * self.weights.shape[1] > 100000:
                # For large operations, use our optimized functions
                output = fast_matrix_multiply(inputs_contig, weights_contig)
                output = fast_add_bias(output, self.biases)
            else:
                # For smaller operations, use NumPy's optimized BLAS
                output = np.dot(inputs_contig, weights_contig) + self.biases
        else:
            # Fallback to NumPy operations
            output = np.dot(inputs, self.weights) + self.biases
        
        return output
    
    def backward(self, grad_output):
        """
        Optimized backward pass through dense layer.
        
        Args:
            grad_output: Gradient of loss w.r.t. layer output, shape (batch_size, output_size)
            
        Returns:
            Gradient of loss w.r.t. layer input, shape (batch_size, input_size)
        """
        # Ensure grad_output is float32
        grad_output = grad_output.astype(np.float32)
        
        if self.use_fast_ops:
            # Ensure arrays are contiguous
            grad_output_contig = np.ascontiguousarray(grad_output)
            inputs_contig = np.ascontiguousarray(self.last_inputs)
            weights_contig = np.ascontiguousarray(self.weights)
            
            # Use optimized gradient computations for large operations
            if grad_output.size > 10000:
                self.grad_weights = fast_gradient_weights(inputs_contig, grad_output_contig)
                self.grad_biases = fast_gradient_bias(grad_output_contig)
                grad_input = fast_matrix_multiply(grad_output_contig, weights_contig.T)
            else:
                # Use NumPy for smaller operations
                self.grad_weights = np.dot(inputs_contig.T, grad_output_contig)
                self.grad_biases = np.sum(grad_output_contig, axis=0, keepdims=True)
                grad_input = np.dot(grad_output_contig, weights_contig.T)
        else:
            # Fallback to NumPy operations
            self.grad_weights = np.dot(self.last_inputs.T, grad_output)
            self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
            grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input
    
    def get_parameters(self):
        """Get weights and biases."""
        return [self.weights, self.biases]
    
    def get_gradients(self):
        """Get weight and bias gradients."""
        return [self.grad_weights, self.grad_biases]


class OptimizedReLU(Layer):
    """
    Optimized ReLU activation function using Numba JIT compilation.
    """
    
    def __init__(self, use_fast_ops=True):
        """
        Initialize Optimized ReLU activation.
        
        Args:
            use_fast_ops: Whether to use JIT-compiled operations (default: True)
        """
        self.use_fast_ops = use_fast_ops
        self.last_inputs = None
    
    def forward(self, inputs):
        """
        Optimized forward pass through ReLU.
        
        Args:
            inputs: Input data of any shape
            
        Returns:
            Output with same shape as input, with negative values set to 0
        """
        # Ensure inputs are float32
        inputs = inputs.astype(np.float32)
        
        # Store inputs for backward pass
        self.last_inputs = inputs.copy()
        
        if self.use_fast_ops:
            output = fast_relu_forward(inputs)
        else:
            output = np.maximum(0, inputs)
        
        return output
    
    def backward(self, grad_output):
        """
        Optimized backward pass through ReLU.
        
        Args:
            grad_output: Gradient of loss w.r.t. layer output
            
        Returns:
            Gradient of loss w.r.t. layer input
        """
        # Ensure grad_output is float32
        grad_output = grad_output.astype(np.float32)
        
        if self.use_fast_ops:
            grad_input = fast_relu_backward(grad_output, self.last_inputs)
        else:
            relu_mask = (self.last_inputs > 0).astype(np.float32)
            grad_input = grad_output * relu_mask
        
        return grad_input


class OptimizedSigmoid(Layer):
    """
    Optimized Sigmoid activation function using Numba JIT compilation.
    """
    
    def __init__(self, use_fast_ops=True):
        """
        Initialize Optimized Sigmoid activation.
        
        Args:
            use_fast_ops: Whether to use JIT-compiled operations (default: True)
        """
        self.use_fast_ops = use_fast_ops
        self.last_output = None
    
    def forward(self, inputs):
        """
        Optimized forward pass through Sigmoid.
        
        Args:
            inputs: Input data of any shape
            
        Returns:
            Output with same shape as input, values in range (0, 1)
        """
        # Ensure inputs are float32
        inputs = inputs.astype(np.float32)
        
        if self.use_fast_ops:
            output = fast_sigmoid_forward(inputs)
        else:
            # Fallback with numerical stability
            clipped_inputs = np.clip(inputs, -500, 500)
            output = 1.0 / (1.0 + np.exp(-clipped_inputs))
        
        # Store output for backward pass
        self.last_output = output.copy()
        
        return output
    
    def backward(self, grad_output):
        """
        Optimized backward pass through Sigmoid.
        
        Args:
            grad_output: Gradient of loss w.r.t. layer output
            
        Returns:
            Gradient of loss w.r.t. layer input
        """
        # Ensure grad_output is float32
        grad_output = grad_output.astype(np.float32)
        
        if self.use_fast_ops:
            grad_input = fast_sigmoid_backward(grad_output, self.last_output)
        else:
            sigmoid_derivative = self.last_output * (1.0 - self.last_output)
            grad_input = grad_output * sigmoid_derivative
        
        return grad_input


class OptimizedTanh(Layer):
    """
    Optimized Tanh activation function using Numba JIT compilation.
    """
    
    def __init__(self, use_fast_ops=True):
        """
        Initialize Optimized Tanh activation.
        
        Args:
            use_fast_ops: Whether to use JIT-compiled operations (default: True)
        """
        self.use_fast_ops = use_fast_ops
        self.last_output = None
    
    def forward(self, inputs):
        """
        Optimized forward pass through Tanh.
        
        Args:
            inputs: Input data of any shape
            
        Returns:
            Output with same shape as input, values in range (-1, 1)
        """
        # Ensure inputs are float32
        inputs = inputs.astype(np.float32)
        
        if self.use_fast_ops:
            output = fast_tanh_forward(inputs)
        else:
            output = np.tanh(inputs)
        
        # Store output for backward pass
        self.last_output = output.copy()
        
        return output
    
    def backward(self, grad_output):
        """
        Optimized backward pass through Tanh.
        
        Args:
            grad_output: Gradient of loss w.r.t. layer output
            
        Returns:
            Gradient of loss w.r.t. layer input
        """
        # Ensure grad_output is float32
        grad_output = grad_output.astype(np.float32)
        
        if self.use_fast_ops:
            grad_input = fast_tanh_backward(grad_output, self.last_output)
        else:
            tanh_derivative = 1.0 - np.square(self.last_output)
            grad_input = grad_output * tanh_derivative
        
        return grad_input


class OptimizedSoftmax(Layer):
    """
    Optimized Softmax activation function.
    
    Note: Softmax is more complex to optimize due to its cross-dependencies,
    so this version focuses on numerical stability and memory efficiency.
    """
    
    def __init__(self, use_fast_ops=True):
        """
        Initialize Optimized Softmax activation.
        
        Args:
            use_fast_ops: Whether to use optimized operations (default: True)
        """
        self.use_fast_ops = use_fast_ops
        self.last_output = None
    
    def forward(self, inputs):
        """
        Optimized forward pass through Softmax.
        
        Args:
            inputs: Input data of shape (batch_size, num_classes)
            
        Returns:
            Output with same shape as input, values sum to 1 along last axis
        """
        # Ensure inputs are float32
        inputs = inputs.astype(np.float32)
        
        # Subtract max for numerical stability
        shifted_inputs = inputs - np.max(inputs, axis=-1, keepdims=True)
        
        # Compute exponentials
        exp_values = np.exp(shifted_inputs)
        
        # Compute softmax
        output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        
        # Store output for backward pass
        self.last_output = output.copy()
        
        return output
    
    def backward(self, grad_output):
        """
        Optimized backward pass through Softmax.
        
        Args:
            grad_output: Gradient of loss w.r.t. layer output
            
        Returns:
            Gradient of loss w.r.t. layer input
        """
        # Ensure grad_output is float32
        grad_output = grad_output.astype(np.float32)
        
        batch_size = self.last_output.shape[0]
        grad_input = np.zeros_like(grad_output)
        
        # Vectorized computation for better performance
        for i in range(batch_size):
            s = self.last_output[i:i+1]
            sum_term = np.sum(grad_output[i:i+1] * s, axis=-1, keepdims=True)
            grad_input[i:i+1] = s * (grad_output[i:i+1] - sum_term)
        
        return grad_input


# =============================================================================
# Batch Processing Layers
# =============================================================================

class BatchNormalization(Layer):
    """
    Batch normalization layer for improved training stability and speed.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Initialize Batch Normalization layer.
        
        Args:
            num_features: Number of features (channels)
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features, dtype=np.float32)  # Scale
        self.beta = np.zeros(num_features, dtype=np.float32)  # Shift
        
        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        
        # Training mode flag
        self.training = True
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, inputs):
        """
        Forward pass through batch normalization.
        
        Args:
            inputs: Input data of shape (batch_size, num_features)
            
        Returns:
            Normalized output
        """
        inputs = inputs.astype(np.float32)
        
        if self.training:
            # Compute batch statistics
            batch_mean = np.mean(inputs, axis=0)
            batch_var = np.var(inputs, axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Normalize
            normalized = (inputs - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Cache for backward pass
            self.cache = (inputs, normalized, batch_mean, batch_var)
        else:
            # Use running statistics for inference
            normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        output = self.gamma * normalized + self.beta
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass through batch normalization.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient w.r.t. inputs
        """
        if not self.training or self.cache is None:
            # Simple pass-through for inference mode
            return grad_output
        
        inputs, normalized, batch_mean, batch_var = self.cache
        batch_size = inputs.shape[0]
        
        # Gradients w.r.t. gamma and beta
        self.grad_gamma = np.sum(grad_output * normalized, axis=0)
        self.grad_beta = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t. normalized input
        grad_normalized = grad_output * self.gamma
        
        # Gradient w.r.t. variance
        grad_var = np.sum(grad_normalized * (inputs - batch_mean), axis=0) * \
                   (-0.5) * np.power(batch_var + self.eps, -1.5)
        
        # Gradient w.r.t. mean
        grad_mean = np.sum(grad_normalized * (-1.0 / np.sqrt(batch_var + self.eps)), axis=0) + \
                    grad_var * np.sum(-2.0 * (inputs - batch_mean), axis=0) / batch_size
        
        # Gradient w.r.t. input
        grad_input = grad_normalized / np.sqrt(batch_var + self.eps) + \
                     grad_var * 2.0 * (inputs - batch_mean) / batch_size + \
                     grad_mean / batch_size
        
        return grad_input
    
    def get_parameters(self):
        """Get learnable parameters."""
        return [self.gamma, self.beta]
    
    def get_gradients(self):
        """Get parameter gradients."""
        return [getattr(self, 'grad_gamma', np.zeros_like(self.gamma)),
                getattr(self, 'grad_beta', np.zeros_like(self.beta))]
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False


# =============================================================================
# Factory Functions
# =============================================================================

def create_optimized_dense(input_size, output_size, use_fast_ops=True):
    """
    Factory function to create an optimized dense layer.
    
    Args:
        input_size: Number of input features
        output_size: Number of output features
        use_fast_ops: Whether to use JIT-compiled operations
        
    Returns:
        OptimizedDense layer
    """
    return OptimizedDense(input_size, output_size, use_fast_ops)


def create_optimized_activation(activation_type, use_fast_ops=True):
    """
    Factory function to create optimized activation layers.
    
    Args:
        activation_type: Type of activation ('relu', 'sigmoid', 'tanh', 'softmax')
        use_fast_ops: Whether to use JIT-compiled operations
        
    Returns:
        Optimized activation layer
    """
    activation_map = {
        'relu': OptimizedReLU,
        'sigmoid': OptimizedSigmoid,
        'tanh': OptimizedTanh,
        'softmax': OptimizedSoftmax
    }
    
    if activation_type.lower() not in activation_map:
        raise ValueError(f"Unsupported activation type: {activation_type}")
    
    return activation_map[activation_type.lower()](use_fast_ops)


def benchmark_layer_performance(layer_class, input_shape, num_iterations=100):
    """
    Benchmark the performance of a layer implementation.
    
    Args:
        layer_class: Layer class to benchmark
        input_shape: Shape of input data (batch_size, features)
        num_iterations: Number of iterations for timing
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Create layer instance
    if hasattr(layer_class, '__name__') and 'Dense' in layer_class.__name__:
        layer = layer_class(input_shape[1], input_shape[1])
    else:
        layer = layer_class()
    
    # Generate random input data
    inputs = np.random.randn(*input_shape).astype(np.float32)
    
    # Warm-up runs
    for _ in range(10):
        output = layer.forward(inputs)
        if hasattr(layer, 'backward'):
            grad_output = np.random.randn(*output.shape).astype(np.float32)
            layer.backward(grad_output)
    
    # Time forward pass
    start_time = time.time()
    for _ in range(num_iterations):
        output = layer.forward(inputs)
    forward_time = (time.time() - start_time) / num_iterations
    
    # Time backward pass
    grad_output = np.random.randn(*output.shape).astype(np.float32)
    start_time = time.time()
    for _ in range(num_iterations):
        layer.backward(grad_output)
    backward_time = (time.time() - start_time) / num_iterations
    
    return {
        'forward_time': forward_time,
        'backward_time': backward_time,
        'total_time': forward_time + backward_time,
        'layer_type': layer_class.__name__
    }