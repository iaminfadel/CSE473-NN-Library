"""
Optimization algorithms for neural network training.

This module implements various optimization algorithms used to update
neural network parameters during training.
"""

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    Implements the basic SGD parameter update rule:
    parameter = parameter - learning_rate * gradient
    
    This is the fundamental optimization algorithm that updates parameters
    in the direction opposite to the gradient to minimize the loss function.
    """
    
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate (float): Learning rate for parameter updates.
                                 Controls the step size in gradient descent.
                                 Default is 0.01.
            weight_decay (float): L2 regularization coefficient (weight decay).
                                Controls the strength of regularization.
                                Default is 0.0 (no regularization).
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def step(self, parameters, gradients):
        """
        Perform one optimization step by updating parameters.
        
        Args:
            parameters (list): List of parameter arrays to update
            gradients (list): List of gradient arrays corresponding to parameters
            
        Note:
            This method modifies the parameters in-place.
            The parameters and gradients lists must have the same length
            and corresponding elements must have the same shape.
            
            If weight_decay > 0, applies L2 regularization by adding
            weight_decay * parameter to the gradient before updating.
        """
        # Ensure parameters and gradients lists have the same length
        if len(parameters) != len(gradients):
            raise ValueError(f"Number of parameters ({len(parameters)}) must match "
                           f"number of gradients ({len(gradients)})")
        
        # Update each parameter using SGD rule with optional weight decay
        for param, grad in zip(parameters, gradients):
            if param.shape != grad.shape:
                raise ValueError(f"Parameter shape {param.shape} must match "
                               f"gradient shape {grad.shape}")
            
            # Apply L2 regularization (weight decay) if specified
            # Effective gradient = gradient + weight_decay * parameter
            effective_grad = grad + self.weight_decay * param
            
            # SGD update rule: parameter = parameter - learning_rate * effective_gradient
            param -= self.learning_rate * effective_grad
    
    def zero_gradients(self, gradients):
        """
        Zero out all gradients.
        
        Args:
            gradients (list): List of gradient arrays to zero out
            
        Note:
            This method modifies the gradients in-place.
        """
        for grad in gradients:
            grad.fill(0.0)
    
    def get_learning_rate(self):
        """
        Get the current learning rate.
        
        Returns:
            float: Current learning rate
        """
        return self.learning_rate
    
    def set_learning_rate(self, learning_rate):
        """
        Set a new learning rate.
        
        Args:
            learning_rate (float): New learning rate value
        """
        self.learning_rate = learning_rate