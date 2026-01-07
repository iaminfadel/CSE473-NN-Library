"""
Optimized loss function implementations using Numba JIT compilation.

This module provides high-performance versions of loss functions
with significant speed improvements over the base implementations.
"""

import numpy as np
from .optimized_ops import (
    fast_mse_loss, fast_mse_gradient,
    fast_bce_with_logits_loss, fast_bce_with_logits_gradient
)


class OptimizedMSELoss:
    """
    Optimized Mean Squared Error loss function using Numba JIT compilation.
    
    Provides significant performance improvements over the base MSELoss
    implementation through JIT compilation and parallel processing.
    """
    
    def __init__(self, use_fast_ops=True):
        """
        Initialize Optimized MSE loss function.
        
        Args:
            use_fast_ops: Whether to use JIT-compiled operations (default: True)
        """
        self.use_fast_ops = use_fast_ops
    
    def forward(self, predictions, targets):
        """
        Optimized MSE loss computation.
        
        Args:
            predictions (np.ndarray): Model predictions, shape (batch_size, output_dim)
            targets (np.ndarray): Target values, shape (batch_size, output_dim)
            
        Returns:
            float: Mean squared error loss
        """
        # Ensure inputs are numpy arrays and float32 for optimal performance
        predictions = np.asarray(predictions, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        
        if self.use_fast_ops:
            return fast_mse_loss(predictions, targets)
        else:
            # Fallback to NumPy operations
            squared_diff = (predictions - targets) ** 2
            return np.mean(squared_diff)
    
    def backward(self, predictions, targets):
        """
        Optimized MSE gradient computation.
        
        Args:
            predictions (np.ndarray): Model predictions, shape (batch_size, output_dim)
            targets (np.ndarray): Target values, shape (batch_size, output_dim)
            
        Returns:
            np.ndarray: Gradient of loss w.r.t. predictions, same shape as predictions
        """
        # Ensure inputs are numpy arrays and float32
        predictions = np.asarray(predictions, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        
        if self.use_fast_ops:
            return fast_mse_gradient(predictions, targets)
        else:
            # Fallback to NumPy operations
            total_elements = predictions.size
            return (2.0 / total_elements) * (predictions - targets)
    
    def __call__(self, predictions, targets):
        """Allow the loss function to be called directly."""
        return self.forward(predictions, targets)


class OptimizedBCEWithLogitsLoss:
    """
    Optimized Binary Cross Entropy with Logits loss function using Numba JIT compilation.
    
    This combines a Sigmoid layer and the BCELoss in one single class with
    JIT compilation for maximum performance and numerical stability.
    """
    
    def __init__(self, use_fast_ops=True):
        """
        Initialize Optimized BCE with logits loss function.
        
        Args:
            use_fast_ops: Whether to use JIT-compiled operations (default: True)
        """
        self.use_fast_ops = use_fast_ops
    
    def forward(self, logits, targets):
        """
        Optimized BCE with logits loss computation.
        
        Args:
            logits (np.ndarray): Raw model outputs (before sigmoid), shape (batch_size, output_dim)
            targets (np.ndarray): Target values (0 or 1), shape (batch_size, output_dim)
            
        Returns:
            float: BCE with logits loss
        """
        # Ensure inputs are numpy arrays and float32
        logits = np.asarray(logits, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        
        if self.use_fast_ops:
            return fast_bce_with_logits_loss(logits, targets)
        else:
            # Fallback to NumPy operations with numerical stability
            max_val = np.clip(logits, 0, None)
            loss_elements = max_val - logits * targets + np.log(1 + np.exp(-np.abs(logits)))
            return np.mean(loss_elements)
    
    def backward(self, logits, targets):
        """
        Optimized BCE with logits gradient computation.
        
        Args:
            logits (np.ndarray): Raw model outputs, shape (batch_size, output_dim)
            targets (np.ndarray): Target values, shape (batch_size, output_dim)
            
        Returns:
            np.ndarray: Gradient of loss w.r.t. logits, same shape as logits
        """
        # Ensure inputs are numpy arrays and float32
        logits = np.asarray(logits, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        
        if self.use_fast_ops:
            return fast_bce_with_logits_gradient(logits, targets)
        else:
            # Fallback to NumPy operations with numerical stability
            sigmoid_logits = np.where(
                logits >= 0,
                1 / (1 + np.exp(-logits)),
                np.exp(logits) / (1 + np.exp(logits))
            )
            total_elements = logits.size
            return (sigmoid_logits - targets) / total_elements
    
    def __call__(self, logits, targets):
        """Allow the loss function to be called directly."""
        return self.forward(logits, targets)


class OptimizedCrossEntropyLoss:
    """
    Optimized Cross Entropy loss function for multi-class classification.
    
    This implementation combines softmax and cross-entropy for numerical stability
    and uses optimized operations for better performance.
    """
    
    def __init__(self, use_fast_ops=True):
        """
        Initialize Optimized Cross Entropy loss function.
        
        Args:
            use_fast_ops: Whether to use optimized operations (default: True)
        """
        self.use_fast_ops = use_fast_ops
    
    def forward(self, logits, targets):
        """
        Optimized cross entropy loss computation.
        
        Args:
            logits (np.ndarray): Raw model outputs, shape (batch_size, num_classes)
            targets (np.ndarray): Target class indices, shape (batch_size,) or one-hot (batch_size, num_classes)
            
        Returns:
            float: Cross entropy loss
        """
        # Ensure inputs are float32
        logits = np.asarray(logits, dtype=np.float32)
        targets = np.asarray(targets)
        
        batch_size = logits.shape[0]
        
        # Convert targets to one-hot if needed
        if targets.ndim == 1:
            num_classes = logits.shape[1]
            targets_one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
            targets_one_hot[np.arange(batch_size), targets] = 1.0
            targets = targets_one_hot
        else:
            targets = targets.astype(np.float32)
        
        # Compute softmax with numerical stability
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Compute cross entropy loss
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        softmax_probs = np.clip(softmax_probs, epsilon, 1.0 - epsilon)
        
        # Cross entropy: -sum(targets * log(softmax_probs))
        cross_entropy = -np.sum(targets * np.log(softmax_probs), axis=1)
        
        return np.mean(cross_entropy)
    
    def backward(self, logits, targets):
        """
        Optimized cross entropy gradient computation.
        
        Args:
            logits (np.ndarray): Raw model outputs, shape (batch_size, num_classes)
            targets (np.ndarray): Target class indices or one-hot vectors
            
        Returns:
            np.ndarray: Gradient w.r.t. logits
        """
        # Ensure inputs are float32
        logits = np.asarray(logits, dtype=np.float32)
        targets = np.asarray(targets)
        
        batch_size = logits.shape[0]
        
        # Convert targets to one-hot if needed
        if targets.ndim == 1:
            num_classes = logits.shape[1]
            targets_one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
            targets_one_hot[np.arange(batch_size), targets] = 1.0
            targets = targets_one_hot
        else:
            targets = targets.astype(np.float32)
        
        # Compute softmax
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Gradient of cross entropy with softmax: (softmax_probs - targets) / batch_size
        gradient = (softmax_probs - targets) / batch_size
        
        return gradient
    
    def __call__(self, logits, targets):
        """Allow the loss function to be called directly."""
        return self.forward(logits, targets)


class OptimizedHuberLoss:
    """
    Optimized Huber loss function (smooth L1 loss) for robust regression.
    
    Huber loss is less sensitive to outliers than MSE loss while still being
    differentiable everywhere.
    """
    
    def __init__(self, delta=1.0, use_fast_ops=True):
        """
        Initialize Optimized Huber loss function.
        
        Args:
            delta: Threshold parameter for switching between L1 and L2 loss
            use_fast_ops: Whether to use optimized operations (default: True)
        """
        self.delta = delta
        self.use_fast_ops = use_fast_ops
    
    def forward(self, predictions, targets):
        """
        Optimized Huber loss computation.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Target values
            
        Returns:
            float: Huber loss
        """
        # Ensure inputs are float32
        predictions = np.asarray(predictions, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        
        # Compute absolute error
        abs_error = np.abs(predictions - targets)
        
        # Huber loss: 0.5 * error^2 if |error| <= delta, else delta * (|error| - 0.5 * delta)
        quadratic = 0.5 * abs_error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)
        
        huber_loss = np.where(abs_error <= self.delta, quadratic, linear)
        
        return np.mean(huber_loss)
    
    def backward(self, predictions, targets):
        """
        Optimized Huber loss gradient computation.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Target values
            
        Returns:
            np.ndarray: Gradient w.r.t. predictions
        """
        # Ensure inputs are float32
        predictions = np.asarray(predictions, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        
        # Compute error
        error = predictions - targets
        abs_error = np.abs(error)
        
        # Gradient: error if |error| <= delta, else delta * sign(error)
        gradient = np.where(
            abs_error <= self.delta,
            error,
            self.delta * np.sign(error)
        )
        
        # Normalize by number of elements
        return gradient / error.size
    
    def __call__(self, predictions, targets):
        """Allow the loss function to be called directly."""
        return self.forward(predictions, targets)


# =============================================================================
# Factory Functions and Utilities
# =============================================================================

def create_optimized_loss(loss_type, **kwargs):
    """
    Factory function to create optimized loss functions.
    
    Args:
        loss_type: Type of loss ('mse', 'bce_logits', 'cross_entropy', 'huber')
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Optimized loss function instance
    """
    loss_map = {
        'mse': OptimizedMSELoss,
        'bce_logits': OptimizedBCEWithLogitsLoss,
        'cross_entropy': OptimizedCrossEntropyLoss,
        'huber': OptimizedHuberLoss
    }
    
    if loss_type.lower() not in loss_map:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss_map[loss_type.lower()](**kwargs)


def benchmark_loss_performance(loss_class, input_shape, num_iterations=1000):
    """
    Benchmark the performance of a loss function implementation.
    
    Args:
        loss_class: Loss class to benchmark
        input_shape: Shape of input data (batch_size, features)
        num_iterations: Number of iterations for timing
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Create loss instance
    loss_fn = loss_class()
    
    # Generate random data
    predictions = np.random.randn(*input_shape).astype(np.float32)
    targets = np.random.randn(*input_shape).astype(np.float32)
    
    # For classification losses, adjust targets
    if 'BCE' in loss_class.__name__ or 'CrossEntropy' in loss_class.__name__:
        targets = (targets > 0).astype(np.float32)
    
    # Warm-up runs
    for _ in range(10):
        loss_value = loss_fn.forward(predictions, targets)
        gradient = loss_fn.backward(predictions, targets)
    
    # Time forward pass
    start_time = time.time()
    for _ in range(num_iterations):
        loss_value = loss_fn.forward(predictions, targets)
    forward_time = (time.time() - start_time) / num_iterations
    
    # Time backward pass
    start_time = time.time()
    for _ in range(num_iterations):
        gradient = loss_fn.backward(predictions, targets)
    backward_time = (time.time() - start_time) / num_iterations
    
    return {
        'forward_time': forward_time,
        'backward_time': backward_time,
        'total_time': forward_time + backward_time,
        'loss_type': loss_class.__name__,
        'final_loss': float(loss_value)
    }


def compare_loss_implementations(original_loss, optimized_loss, input_shape, tolerance=1e-6):
    """
    Compare original and optimized loss implementations for correctness.
    
    Args:
        original_loss: Original loss function instance
        optimized_loss: Optimized loss function instance
        input_shape: Shape of test data
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dictionary with comparison results
    """
    # Generate test data
    predictions = np.random.randn(*input_shape).astype(np.float32)
    targets = np.random.randn(*input_shape).astype(np.float32)
    
    # Compute losses
    original_loss_val = original_loss.forward(predictions, targets)
    optimized_loss_val = optimized_loss.forward(predictions, targets)
    
    # Compute gradients
    original_grad = original_loss.backward(predictions, targets)
    optimized_grad = optimized_loss.backward(predictions, targets)
    
    # Compare results
    loss_diff = abs(original_loss_val - optimized_loss_val)
    grad_diff = np.max(np.abs(original_grad - optimized_grad))
    
    return {
        'loss_match': loss_diff < tolerance,
        'gradient_match': grad_diff < tolerance,
        'loss_difference': float(loss_diff),
        'gradient_difference': float(grad_diff),
        'original_loss': float(original_loss_val),
        'optimized_loss': float(optimized_loss_val)
    }