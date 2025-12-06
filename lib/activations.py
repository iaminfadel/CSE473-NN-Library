"""
Activation function implementations.

This module contains activation functions implemented as Layer subclasses,
including ReLU, Sigmoid, Tanh, and Softmax.
"""

import numpy as np
from .layers import Layer


class ReLU(Layer):
    """
    Rectified Linear Unit activation function.
    
    Forward: f(x) = max(0, x)
    Backward: f'(x) = 1 if x > 0 else 0
    """
    
    def __init__(self):
        """Initialize ReLU activation."""
        self.last_inputs = None
    
    def forward(self, inputs):
        """
        Forward pass through ReLU.
        
        Args:
            inputs: Input data of any shape
            
        Returns:
            Output with same shape as input, with negative values set to 0
        """
        # Store inputs for backward pass
        self.last_inputs = inputs.copy()
        
        # Apply ReLU: max(0, x)
        output = np.maximum(0, inputs)
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass through ReLU.
        
        Args:
            grad_output: Gradient of loss w.r.t. layer output
            
        Returns:
            Gradient of loss w.r.t. layer input
        """
        # ReLU derivative: 1 if x > 0 else 0
        # Create mask where inputs were positive
        relu_mask = (self.last_inputs > 0).astype(np.float32)
        
        # Apply mask to gradient
        grad_input = grad_output * relu_mask
        
        return grad_input


class Sigmoid(Layer):
    """
    Sigmoid activation function.
    
    Forward: f(x) = 1 / (1 + exp(-x))
    Backward: f'(x) = f(x) * (1 - f(x))
    """
    
    def __init__(self):
        """Initialize Sigmoid activation."""
        self.last_output = None
    
    def forward(self, inputs):
        """
        Forward pass through Sigmoid.
        
        Args:
            inputs: Input data of any shape
            
        Returns:
            Output with same shape as input, values in range (0, 1)
        """
        # Clip inputs to prevent overflow in exp(-x)
        # For numerical stability, clip to reasonable range
        clipped_inputs = np.clip(inputs, -500, 500)
        
        # Apply sigmoid: 1 / (1 + exp(-x))
        output = 1.0 / (1.0 + np.exp(-clipped_inputs))
        
        # Store output for backward pass
        self.last_output = output.copy()
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass through Sigmoid.
        
        Args:
            grad_output: Gradient of loss w.r.t. layer output
            
        Returns:
            Gradient of loss w.r.t. layer input
        """
        # Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        sigmoid_derivative = self.last_output * (1.0 - self.last_output)
        
        # Apply chain rule
        grad_input = grad_output * sigmoid_derivative
        
        return grad_input


class Tanh(Layer):
    """
    Hyperbolic tangent activation function.
    
    Forward: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Backward: f'(x) = 1 - f(x)²
    """
    
    def __init__(self):
        """Initialize Tanh activation."""
        self.last_output = None
    
    def forward(self, inputs):
        """
        Forward pass through Tanh.
        
        Args:
            inputs: Input data of any shape
            
        Returns:
            Output with same shape as input, values in range (-1, 1)
        """
        # Use numpy's tanh for numerical stability
        # This is equivalent to: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        output = np.tanh(inputs)
        
        # Store output for backward pass
        self.last_output = output.copy()
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass through Tanh.
        
        Args:
            grad_output: Gradient of loss w.r.t. layer output
            
        Returns:
            Gradient of loss w.r.t. layer input
        """
        # Tanh derivative: 1 - tanh²(x)
        tanh_derivative = 1.0 - np.square(self.last_output)
        
        # Apply chain rule
        grad_input = grad_output * tanh_derivative
        
        return grad_input


class Softmax(Layer):
    """
    Softmax activation function.
    
    Forward: f(x_i) = exp(x_i) / sum(exp(x_j))
    Backward: Jacobian matrix computation
    """
    
    def __init__(self):
        """Initialize Softmax activation."""
        self.last_output = None
    
    def forward(self, inputs):
        """
        Forward pass through Softmax.
        
        Args:
            inputs: Input data of shape (batch_size, num_classes)
            
        Returns:
            Output with same shape as input, values sum to 1 along last axis
        """
        # Subtract max for numerical stability (prevents overflow)
        # This doesn't change the result due to softmax properties
        shifted_inputs = inputs - np.max(inputs, axis=-1, keepdims=True)
        
        # Compute exponentials
        exp_values = np.exp(shifted_inputs)
        
        # Compute softmax: exp(x_i) / sum(exp(x_j))
        output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        
        # Store output for backward pass
        self.last_output = output.copy()
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass through Softmax.
        
        Args:
            grad_output: Gradient of loss w.r.t. layer output
            
        Returns:
            Gradient of loss w.r.t. layer input
        """
        # Softmax Jacobian computation
        # For softmax, the Jacobian is:
        # J_ij = softmax_i * (δ_ij - softmax_j)
        # where δ_ij is the Kronecker delta
        
        # This can be computed efficiently as:
        # grad_input_i = softmax_i * (grad_output_i - sum_j(grad_output_j * softmax_j))
        
        batch_size = self.last_output.shape[0]
        grad_input = np.zeros_like(grad_output)
        
        for i in range(batch_size):
            # Get softmax output for this sample
            s = self.last_output[i:i+1]  # Keep batch dimension
            
            # Compute the sum term: sum_j(grad_output_j * softmax_j)
            sum_term = np.sum(grad_output[i:i+1] * s, axis=-1, keepdims=True)
            
            # Apply Jacobian: softmax_i * (grad_output_i - sum_term)
            grad_input[i:i+1] = s * (grad_output[i:i+1] - sum_term)
        
        return grad_input