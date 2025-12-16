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


class BCEWithLogitsLoss:
    """
    Binary Cross Entropy with Logits loss function.
    
    This combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss
    as, by combining the operations into one layer, we take advantage of the log-sum-exp trick
    for numerical stability.
    
    The loss is computed as:
    Loss = -1/N * sum(targets * log(sigmoid(logits)) + (1 - targets) * log(1 - sigmoid(logits)))
    
    Using the numerically stable formulation:
    Loss = 1/N * sum(max(logits, 0) - logits * targets + log(1 + exp(-abs(logits))))
    """
    
    def __init__(self):
        """Initialize BCE with logits loss function."""
        pass
    
    def forward(self, logits, targets):
        """
        Compute the BCE with logits loss.
        
        Args:
            logits (np.ndarray): Raw model outputs (before sigmoid), shape (batch_size, output_dim)
            targets (np.ndarray): Target values (0 or 1), shape (batch_size, output_dim)
            
        Returns:
            float: BCE with logits loss
        """
        # Ensure inputs are numpy arrays
        logits = np.array(logits)
        targets = np.array(targets)
        
        # Numerically stable computation of BCE with logits
        # BCE = max(x, 0) - x * z + log(1 + exp(-abs(x)))
        # where x = logits, z = targets
        
        # Clamp logits to prevent overflow in exp
        max_val = np.clip(logits, 0, None)  # max(logits, 0)
        
        # Compute the stable BCE formula
        loss_elements = max_val - logits * targets + np.log(1 + np.exp(-np.abs(logits)))
        
        # Return mean loss
        bce_loss = np.mean(loss_elements)
        
        return bce_loss
    
    def backward(self, logits, targets):
        """
        Compute the gradient of BCE with logits loss with respect to logits.
        
        The gradient is: sigmoid(logits) - targets
        
        Args:
            logits (np.ndarray): Raw model outputs, shape (batch_size, output_dim)
            targets (np.ndarray): Target values, shape (batch_size, output_dim)
            
        Returns:
            np.ndarray: Gradient of loss w.r.t. logits, same shape as logits
        """
        # Ensure inputs are numpy arrays
        logits = np.array(logits)
        targets = np.array(targets)
        
        # Compute sigmoid in a numerically stable way
        # sigmoid(x) = 1 / (1 + exp(-x)) for x >= 0
        # sigmoid(x) = exp(x) / (1 + exp(x)) for x < 0
        sigmoid_logits = np.where(
            logits >= 0,
            1 / (1 + np.exp(-logits)),
            np.exp(logits) / (1 + np.exp(logits))
        )
        
        # Gradient is sigmoid(logits) - targets, normalized by batch size
        total_elements = logits.size
        gradient = (sigmoid_logits - targets) / total_elements
        
        return gradient
    
    def __call__(self, logits, targets):
        """
        Allow the loss function to be called directly.
        
        Args:
            logits (np.ndarray): Raw model outputs
            targets (np.ndarray): Target values
            
        Returns:
            float: BCE with logits loss value
        """
        return self.forward(logits, targets)