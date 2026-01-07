"""
Practical optimized loss function implementations.

This module provides loss functions optimized for real-world performance
rather than theoretical maximum speed.
"""

import numpy as np
from .simple_optimizations import (
    optimized_mse_loss, optimized_mse_gradient,
    optimize_array
)


class PracticalMSELoss:
    """
    Practically optimized Mean Squared Error loss function.
    
    Uses JIT compilation for large arrays, NumPy for small arrays.
    Focuses on float32 precision and memory layout optimizations.
    """
    
    def __init__(self):
        """Initialize Practical MSE loss function."""
        pass
    
    def forward(self, predictions, targets):
        """
        Optimized MSE loss computation.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Target values
            
        Returns:
            float: Mean squared error loss
        """
        # Optimize arrays for better performance
        predictions = optimize_array(predictions, np.float32)
        targets = optimize_array(targets, np.float32)
        
        # Use adaptive optimization based on array size
        return optimized_mse_loss(predictions, targets)
    
    def backward(self, predictions, targets):
        """
        Optimized MSE gradient computation.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Target values
            
        Returns:
            np.ndarray: Gradient of loss w.r.t. predictions
        """
        # Optimize arrays for better performance
        predictions = optimize_array(predictions, np.float32)
        targets = optimize_array(targets, np.float32)
        
        # Use adaptive optimization based on array size
        return optimized_mse_gradient(predictions, targets)
    
    def __call__(self, predictions, targets):
        """Allow the loss function to be called directly."""
        return self.forward(predictions, targets)


class PracticalBCEWithLogitsLoss:
    """
    Practically optimized Binary Cross Entropy with Logits loss function.
    
    Focuses on numerical stability and performance optimizations.
    """
    
    def __init__(self):
        """Initialize Practical BCE with logits loss function."""
        pass
    
    def forward(self, logits, targets):
        """
        Optimized BCE with logits loss computation.
        
        Args:
            logits (np.ndarray): Raw model outputs (before sigmoid)
            targets (np.ndarray): Target values (0 or 1)
            
        Returns:
            float: BCE with logits loss
        """
        # Optimize arrays
        logits = optimize_array(logits, np.float32)
        targets = optimize_array(targets, np.float32)
        
        # Numerically stable computation
        # Use the log-sum-exp trick for stability
        max_val = np.maximum(logits, 0)
        loss_elements = max_val - logits * targets + np.log(1 + np.exp(-np.abs(logits)))
        
        return np.mean(loss_elements)
    
    def backward(self, logits, targets):
        """
        Optimized BCE with logits gradient computation.
        
        Args:
            logits (np.ndarray): Raw model outputs
            targets (np.ndarray): Target values
            
        Returns:
            np.ndarray: Gradient of loss w.r.t. logits
        """
        # Optimize arrays
        logits = optimize_array(logits, np.float32)
        targets = optimize_array(targets, np.float32)
        
        # Compute sigmoid in numerically stable way
        sigmoid_logits = np.where(
            logits >= 0,
            1 / (1 + np.exp(-logits)),
            np.exp(logits) / (1 + np.exp(logits))
        )
        
        # Gradient: (sigmoid(logits) - targets) / batch_size
        return (sigmoid_logits - targets) / logits.size
    
    def __call__(self, logits, targets):
        """Allow the loss function to be called directly."""
        return self.forward(logits, targets)


class PracticalCrossEntropyLoss:
    """
    Practically optimized Cross Entropy loss function for multi-class classification.
    
    Combines softmax and cross-entropy for numerical stability.
    """
    
    def __init__(self):
        """Initialize Practical Cross Entropy loss function."""
        pass
    
    def forward(self, logits, targets):
        """
        Optimized cross entropy loss computation.
        
        Args:
            logits (np.ndarray): Raw model outputs, shape (batch_size, num_classes)
            targets (np.ndarray): Target class indices or one-hot vectors
            
        Returns:
            float: Cross entropy loss
        """
        # Optimize arrays
        logits = optimize_array(logits, np.float32)
        targets = np.asarray(targets)
        
        batch_size = logits.shape[0]
        
        # Convert targets to one-hot if needed
        if targets.ndim == 1:
            num_classes = logits.shape[1]
            targets_one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
            targets_one_hot[np.arange(batch_size), targets] = 1.0
            targets = targets_one_hot
        else:
            targets = optimize_array(targets, np.float32)
        
        # Compute softmax with numerical stability
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Compute cross entropy loss
        epsilon = 1e-15
        softmax_probs = np.clip(softmax_probs, epsilon, 1.0 - epsilon)
        
        cross_entropy = -np.sum(targets * np.log(softmax_probs), axis=1)
        
        return np.mean(cross_entropy)
    
    def backward(self, logits, targets):
        """
        Optimized cross entropy gradient computation.
        
        Args:
            logits (np.ndarray): Raw model outputs
            targets (np.ndarray): Target class indices or one-hot vectors
            
        Returns:
            np.ndarray: Gradient w.r.t. logits
        """
        # Optimize arrays
        logits = optimize_array(logits, np.float32)
        targets = np.asarray(targets)
        
        batch_size = logits.shape[0]
        
        # Convert targets to one-hot if needed
        if targets.ndim == 1:
            num_classes = logits.shape[1]
            targets_one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
            targets_one_hot[np.arange(batch_size), targets] = 1.0
            targets = targets_one_hot
        else:
            targets = optimize_array(targets, np.float32)
        
        # Compute softmax
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Gradient: (softmax_probs - targets) / batch_size
        return (softmax_probs - targets) / batch_size
    
    def __call__(self, logits, targets):
        """Allow the loss function to be called directly."""
        return self.forward(logits, targets)


# =============================================================================
# Factory Functions
# =============================================================================

def create_practical_loss(loss_type):
    """
    Factory function to create practical loss functions.
    
    Args:
        loss_type: Type of loss ('mse', 'bce_logits', 'cross_entropy')
        
    Returns:
        Practical loss function instance
    """
    loss_map = {
        'mse': PracticalMSELoss,
        'bce_logits': PracticalBCEWithLogitsLoss,
        'cross_entropy': PracticalCrossEntropyLoss
    }
    
    if loss_type.lower() not in loss_map:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss_map[loss_type.lower()]()


def benchmark_practical_losses(loss_type, input_shape, iterations=1000):
    """
    Benchmark practical loss functions against original implementations.
    
    Args:
        loss_type: Type of loss to benchmark
        input_shape: Shape of input data
        iterations: Number of iterations for timing
        
    Returns:
        Dictionary with timing results
    """
    import time
    from .losses import MSELoss
    
    # Generate test data
    predictions = np.random.randn(*input_shape).astype(np.float32)
    targets = np.random.randn(*input_shape).astype(np.float32)
    
    if loss_type == 'mse':
        original_loss = MSELoss()
        practical_loss = PracticalMSELoss()
    else:
        raise ValueError(f"Benchmark not implemented for {loss_type}")
    
    # Warm-up runs
    for _ in range(10):
        original_loss.forward(predictions, targets)
        original_loss.backward(predictions, targets)
        practical_loss.forward(predictions, targets)
        practical_loss.backward(predictions, targets)
    
    # Time original implementation
    start_time = time.time()
    for _ in range(iterations):
        loss_value = original_loss.forward(predictions, targets)
        gradient = original_loss.backward(predictions, targets)
    original_time = time.time() - start_time
    
    # Time practical implementation
    start_time = time.time()
    for _ in range(iterations):
        loss_value = practical_loss.forward(predictions, targets)
        gradient = practical_loss.backward(predictions, targets)
    practical_time = time.time() - start_time
    
    speedup = original_time / practical_time
    
    return {
        'loss_type': loss_type,
        'input_shape': input_shape,
        'original_time': original_time,
        'practical_time': practical_time,
        'speedup': speedup,
        'iterations': iterations
    }