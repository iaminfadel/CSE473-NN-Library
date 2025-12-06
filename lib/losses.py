"""
Loss functions for neural network training.

This module implements various loss functions used in neural network training,
including their forward pass (loss computation) and backward pass (gradient computation).
"""

import numpy as np


class MSELoss:
    """
    Mean Squared Error loss function.
    
    Computes the mean squared error between predictions and targets:
    Loss = (1/N) * sum((predictions - targets)²)
    
    The gradient with respect to predictions is:
    ∂Loss/∂predictions = 2/N * (predictions - targets)
    """
    
    def __init__(self):
        """Initialize MSE loss function."""
        pass
    
    def forward(self, predictions, targets):
        """
        Compute the MSE loss.
        
        Args:
            predictions (np.ndarray): Model predictions, shape (batch_size, output_dim)
            targets (np.ndarray): Target values, shape (batch_size, output_dim)
            
        Returns:
            float: Mean squared error loss
        """
        # Ensure inputs are numpy arrays
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Compute squared differences
        squared_diff = (predictions - targets) ** 2
        
        # Compute mean over all elements
        mse_loss = np.mean(squared_diff)
        
        return mse_loss
    
    def backward(self, predictions, targets):
        """
        Compute the gradient of MSE loss with respect to predictions.
        
        Args:
            predictions (np.ndarray): Model predictions, shape (batch_size, output_dim)
            targets (np.ndarray): Target values, shape (batch_size, output_dim)
        Returns:
            np.ndarray: Gradient of loss w.r.t. predictions, same shape as predictions
        """
        # Ensure inputs are numpy arrays
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Get total number of elements for normalization (consistent with forward pass)
        total_elements = predictions.size
        
        # Compute gradient: 2/N * (predictions - targets)
        # where N is total number of elements (consistent with np.mean in forward pass)
        gradient = (2.0 / total_elements) * (predictions - targets)
        
        return gradient
    
    def __call__(self, predictions, targets):
        """
        Allow the loss function to be called directly.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Target values
            
        Returns:
            float: MSE loss value
        """
        return self.forward(predictions, targets)