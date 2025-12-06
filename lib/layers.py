"""
Neural network layer implementations.

This module contains the base Layer class and specific layer implementations
like Dense (fully connected) layers.
"""

import numpy as np


class Layer:
    """Abstract base class for all neural network layers."""
    
    def forward(self, inputs):
        """
        Forward propagation through the layer.
        
        Args:
            inputs: Input data to the layer
            
        Returns:
            Output of the layer
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def backward(self, grad_output):
        """
        Backward propagation through the layer.
        
        Args:
            grad_output: Gradient of loss with respect to layer output
            
        Returns:
            Gradient of loss with respect to layer input
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement backward method")
    
    def get_parameters(self):
        """
        Get trainable parameters of the layer.
        
        Returns:
            List of trainable parameters (empty for layers without parameters)
        """
        return []
    
    def get_gradients(self):
        """
        Get gradients of trainable parameters.
        
        Returns:
            List of parameter gradients (empty for layers without parameters)
        """
        return []


class Dense(Layer):
    """
    Fully connected (dense) layer.
    
    This layer performs a linear transformation: output = input @ weights + bias
    Uses Xavier/Glorot initialization for weights.
    """
    
    def __init__(self, input_size, output_size):
        """
        Initialize Dense layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
        """
        # Xavier/Glorot initialization
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights using Xavier/Glorot initialization
        # Xavier initialization: sqrt(2 / (input_size + output_size))
        xavier_std = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size) * xavier_std
        
        # Initialize biases to zero
        self.biases = np.zeros((1, output_size))
        
        # Storage for gradients
        self.grad_weights = None
        self.grad_biases = None
        
        # Storage for inputs (needed for backward pass)
        self.last_inputs = None
    
    def forward(self, inputs):
        """
        Forward pass through dense layer.
        
        Args:
            inputs: Input data of shape (batch_size, input_size)
            
        Returns:
            Output of shape (batch_size, output_size)
        """
        # Store inputs for backward pass
        self.last_inputs = inputs.copy()
        
        # Compute output: inputs @ weights + biases
        output = np.dot(inputs, self.weights) + self.biases
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass through dense layer.
        
        Args:
            grad_output: Gradient of loss w.r.t. layer output, shape (batch_size, output_size)
            
        Returns:
            Gradient of loss w.r.t. layer input, shape (batch_size, input_size)
        """
        # Compute gradient w.r.t. weights: inputs^T @ grad_output
        self.grad_weights = np.dot(self.last_inputs.T, grad_output)
        
        # Compute gradient w.r.t. biases: sum over batch dimension
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        
        # Compute gradient w.r.t. inputs: grad_output @ weights^T
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input
    
    def get_parameters(self):
        """
        Get weights and biases.
        
        Returns:
            List containing [weights, biases]
        """
        return [self.weights, self.biases]
    
    def get_gradients(self):
        """
        Get weight and bias gradients.
        
        Returns:
            List containing [grad_weights, grad_biases]
        """
        return [self.grad_weights, self.grad_biases]