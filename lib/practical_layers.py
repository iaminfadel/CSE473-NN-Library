"""
Practical optimized layer implementations focusing on real performance gains.

This module provides optimized layers that focus on practical improvements
rather than theoretical optimizations that may not provide real benefits.
"""

import numpy as np
from .layers import Layer
from .simple_optimizations import (
    optimized_relu_forward, optimized_relu_backward,
    optimized_sigmoid_forward, optimized_sigmoid_backward,
    optimize_array, adaptive_matrix_multiply
)


class PracticalDense(Layer):
    """
    Practically optimized dense layer focusing on real performance improvements.
    
    Key optimizations:
    - Float32 precision for better cache performance
    - Contiguous memory layout
    - Optimal data type handling
    - Smart fallback to NumPy BLAS for matrix operations
    """
    
    def __init__(self, input_size, output_size):
        """
        Initialize Practical Dense layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights using Xavier/Glorot initialization with float32
        xavier_std = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size).astype(np.float32) * xavier_std
        
        # Initialize biases to zero with float32
        self.biases = np.zeros((1, output_size), dtype=np.float32)
        
        # Ensure weights and biases are contiguous
        self.weights = np.ascontiguousarray(self.weights)
        self.biases = np.ascontiguousarray(self.biases)
        
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
        # Optimize input array (convert to float32 and ensure contiguous)
        inputs = optimize_array(inputs, np.float32)
        
        # Store inputs for backward pass
        self.last_inputs = inputs.copy()
        
        # Use NumPy's optimized BLAS for matrix multiplication
        # This is typically faster than custom implementations for most sizes
        output = adaptive_matrix_multiply(inputs, self.weights)
        
        # Add bias (broadcasting is very efficient in NumPy)
        output = output + self.biases
        
        return output
    
    def backward(self, grad_output):
        """
        Optimized backward pass through dense layer.
        
        Args:
            grad_output: Gradient of loss w.r.t. layer output
            
        Returns:
            Gradient of loss w.r.t. layer input
        """
        # Optimize gradient array
        grad_output = optimize_array(grad_output, np.float32)
        
        # Compute gradients using NumPy's optimized operations
        self.grad_weights = adaptive_matrix_multiply(self.last_inputs.T, grad_output)
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        
        # Compute gradient w.r.t. inputs
        grad_input = adaptive_matrix_multiply(grad_output, self.weights.T)
        
        return grad_input
    
    def get_parameters(self):
        """Get weights and biases."""
        return [self.weights, self.biases]
    
    def get_gradients(self):
        """Get weight and bias gradients."""
        return [self.grad_weights, self.grad_biases]


class PracticalReLU(Layer):
    """
    Practically optimized ReLU activation function.
    
    Uses JIT compilation for large arrays, NumPy for small arrays.
    """
    
    def __init__(self):
        """Initialize Practical ReLU activation."""
        self.last_inputs = None
    
    def forward(self, inputs):
        """
        Optimized forward pass through ReLU.
        
        Args:
            inputs: Input data of any shape
            
        Returns:
            Output with same shape as input, with negative values set to 0
        """
        # Optimize input array
        inputs = optimize_array(inputs, np.float32)
        
        # Store inputs for backward pass
        self.last_inputs = inputs.copy()
        
        # Use adaptive optimization based on array size
        output = optimized_relu_forward(inputs)
        
        return output
    
    def backward(self, grad_output):
        """
        Optimized backward pass through ReLU.
        
        Args:
            grad_output: Gradient of loss w.r.t. layer output
            
        Returns:
            Gradient of loss w.r.t. layer input
        """
        # Optimize gradient array
        grad_output = optimize_array(grad_output, np.float32)
        
        # Use adaptive optimization
        grad_input = optimized_relu_backward(grad_output, self.last_inputs)
        
        return grad_input


class PracticalSigmoid(Layer):
    """
    Practically optimized Sigmoid activation function.
    
    Uses JIT compilation for large arrays, NumPy for small arrays.
    """
    
    def __init__(self):
        """Initialize Practical Sigmoid activation."""
        self.last_output = None
    
    def forward(self, inputs):
        """
        Optimized forward pass through Sigmoid.
        
        Args:
            inputs: Input data of any shape
            
        Returns:
            Output with same shape as input, values in range (0, 1)
        """
        # Optimize input array
        inputs = optimize_array(inputs, np.float32)
        
        # Use adaptive optimization
        output = optimized_sigmoid_forward(inputs)
        
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
        # Optimize gradient array
        grad_output = optimize_array(grad_output, np.float32)
        
        # Use adaptive optimization
        grad_input = optimized_sigmoid_backward(grad_output, self.last_output)
        
        return grad_input


class PracticalTanh(Layer):
    """
    Practically optimized Tanh activation function.
    
    For Tanh, NumPy's implementation is typically very efficient,
    so we focus on data type and memory layout optimizations.
    """
    
    def __init__(self):
        """Initialize Practical Tanh activation."""
        self.last_output = None
    
    def forward(self, inputs):
        """
        Optimized forward pass through Tanh.
        
        Args:
            inputs: Input data of any shape
            
        Returns:
            Output with same shape as input, values in range (-1, 1)
        """
        # Optimize input array
        inputs = optimize_array(inputs, np.float32)
        
        # NumPy's tanh is highly optimized
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
        # Optimize gradient array
        grad_output = optimize_array(grad_output, np.float32)
        
        # Tanh derivative: 1 - tanh²(x)
        tanh_derivative = 1.0 - np.square(self.last_output)
        grad_input = grad_output * tanh_derivative
        
        return grad_input


# =============================================================================
# Factory Functions
# =============================================================================

def create_practical_dense(input_size, output_size):
    """
    Factory function to create a practical dense layer.
    
    Args:
        input_size: Number of input features
        output_size: Number of output features
        
    Returns:
        PracticalDense layer
    """
    return PracticalDense(input_size, output_size)


def create_practical_activation(activation_type):
    """
    Factory function to create practical activation layers.
    
    Args:
        activation_type: Type of activation ('relu', 'sigmoid', 'tanh')
        
    Returns:
        Practical activation layer
    """
    activation_map = {
        'relu': PracticalReLU,
        'sigmoid': PracticalSigmoid,
        'tanh': PracticalTanh
    }
    
    if activation_type.lower() not in activation_map:
        raise ValueError(f"Unsupported activation type: {activation_type}")
    
    return activation_map[activation_type.lower()]()


def benchmark_practical_vs_original(layer_type, input_shape, iterations=100):
    """
    Benchmark practical optimized layers against original implementations.
    
    Args:
        layer_type: Type of layer to benchmark
        input_shape: Shape of input data
        iterations: Number of iterations for timing
        
    Returns:
        Dictionary with timing results
    """
    import time
    from .layers import Dense
    from .activations import ReLU, Sigmoid, Tanh
    
    # Generate test data
    inputs = np.random.randn(*input_shape).astype(np.float32)
    
    if layer_type == 'dense':
        # Original layer
        original_layer = Dense(input_shape[1], input_shape[1])
        
        # Practical layer
        practical_layer = PracticalDense(input_shape[1], input_shape[1])
        
    elif layer_type == 'relu':
        original_layer = ReLU()
        practical_layer = PracticalReLU()
        
    elif layer_type == 'sigmoid':
        original_layer = Sigmoid()
        practical_layer = PracticalSigmoid()
        
    elif layer_type == 'tanh':
        original_layer = Tanh()
        practical_layer = PracticalTanh()
        
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")
    
    # Warm-up runs
    for _ in range(5):
        output_orig = original_layer.forward(inputs)
        output_prac = practical_layer.forward(inputs)
        
        grad_output = np.random.randn(*output_orig.shape).astype(np.float32)
        original_layer.backward(grad_output)
        practical_layer.backward(grad_output)
    
    # Time original implementation
    start_time = time.time()
    for _ in range(iterations):
        output = original_layer.forward(inputs)
        grad_output = np.random.randn(*output.shape).astype(np.float32)
        original_layer.backward(grad_output)
    original_time = time.time() - start_time
    
    # Time practical implementation
    start_time = time.time()
    for _ in range(iterations):
        output = practical_layer.forward(inputs)
        grad_output = np.random.randn(*output.shape).astype(np.float32)
        practical_layer.backward(grad_output)
    practical_time = time.time() - start_time
    
    speedup = original_time / practical_time
    
    return {
        'layer_type': layer_type,
        'input_shape': input_shape,
        'original_time': original_time,
        'practical_time': practical_time,
        'speedup': speedup,
        'iterations': iterations
    }