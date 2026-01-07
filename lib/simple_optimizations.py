"""
Simple and effective optimizations for the neural network library.

This module focuses on practical optimizations that provide real performance
improvements without the overhead of complex JIT compilation for small operations.
"""

import numpy as np
from numba import jit
import warnings

# Disable Numba warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='numba')


# =============================================================================
# Element-wise Operations (where Numba excels)
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def fast_relu_forward_impl(x_flat, result_flat):
    """Fast ReLU implementation for flattened arrays."""
    for i in range(x_flat.size):
        result_flat[i] = max(0.0, x_flat[i])


@jit(nopython=True, cache=True, fastmath=True)
def fast_relu_backward_impl(grad_flat, input_flat, result_flat):
    """Fast ReLU backward implementation for flattened arrays."""
    for i in range(grad_flat.size):
        if input_flat[i] > 0.0:
            result_flat[i] = grad_flat[i]
        else:
            result_flat[i] = 0.0


@jit(nopython=True, cache=True, fastmath=True)
def fast_sigmoid_forward_impl(x_flat, result_flat):
    """Fast sigmoid implementation for flattened arrays."""
    for i in range(x_flat.size):
        val = x_flat[i]
        # Clip for numerical stability
        if val > 500.0:
            val = 500.0
        elif val < -500.0:
            val = -500.0
        
        if val >= 0:
            exp_neg = np.exp(-val)
            result_flat[i] = 1.0 / (1.0 + exp_neg)
        else:
            exp_pos = np.exp(val)
            result_flat[i] = exp_pos / (1.0 + exp_pos)


@jit(nopython=True, cache=True, fastmath=True)
def fast_sigmoid_backward_impl(grad_flat, sigmoid_flat, result_flat):
    """Fast sigmoid backward implementation for flattened arrays."""
    for i in range(grad_flat.size):
        s = sigmoid_flat[i]
        result_flat[i] = grad_flat[i] * s * (1.0 - s)


@jit(nopython=True, cache=True, fastmath=True)
def fast_mse_loss_impl(pred_flat, target_flat):
    """Fast MSE loss computation for flattened arrays."""
    total_loss = 0.0
    n_elements = pred_flat.size
    
    for i in range(n_elements):
        diff = pred_flat[i] - target_flat[i]
        total_loss += diff * diff
    
    return total_loss / n_elements


@jit(nopython=True, cache=True, fastmath=True)
def fast_mse_gradient_impl(pred_flat, target_flat, result_flat):
    """Fast MSE gradient computation for flattened arrays."""
    n_elements = pred_flat.size
    scale = 2.0 / n_elements
    
    for i in range(n_elements):
        result_flat[i] = scale * (pred_flat[i] - target_flat[i])


# =============================================================================
# High-level Optimized Functions
# =============================================================================

def optimized_relu_forward(x):
    """
    Optimized ReLU forward pass.
    
    Args:
        x: Input array
        
    Returns:
        ReLU output
    """
    # Only use JIT for larger arrays where the overhead is worth it
    if x.size > 1000:
        result = np.empty_like(x)
        fast_relu_forward_impl(x.flatten(), result.flatten())
        return result
    else:
        return np.maximum(0, x)


def optimized_relu_backward(grad_output, inputs):
    """
    Optimized ReLU backward pass.
    
    Args:
        grad_output: Gradient from next layer
        inputs: Original inputs to ReLU
        
    Returns:
        Gradient w.r.t. inputs
    """
    if grad_output.size > 1000:
        result = np.empty_like(grad_output)
        fast_relu_backward_impl(grad_output.flatten(), inputs.flatten(), result.flatten())
        return result
    else:
        return grad_output * (inputs > 0)


def optimized_sigmoid_forward(x):
    """
    Optimized sigmoid forward pass.
    
    Args:
        x: Input array
        
    Returns:
        Sigmoid output
    """
    if x.size > 1000:
        result = np.empty_like(x)
        fast_sigmoid_forward_impl(x.flatten(), result.flatten())
        return result
    else:
        # Use NumPy for small arrays
        clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-clipped))


def optimized_sigmoid_backward(grad_output, sigmoid_output):
    """
    Optimized sigmoid backward pass.
    
    Args:
        grad_output: Gradient from next layer
        sigmoid_output: Output from sigmoid forward pass
        
    Returns:
        Gradient w.r.t. inputs
    """
    if grad_output.size > 1000:
        result = np.empty_like(grad_output)
        fast_sigmoid_backward_impl(grad_output.flatten(), sigmoid_output.flatten(), result.flatten())
        return result
    else:
        return grad_output * sigmoid_output * (1.0 - sigmoid_output)


def optimized_mse_loss(predictions, targets):
    """
    Optimized MSE loss computation.
    
    Args:
        predictions: Model predictions
        targets: Target values
        
    Returns:
        MSE loss value
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    
    if predictions.size > 1000:
        return fast_mse_loss_impl(predictions.flatten(), targets.flatten())
    else:
        return np.mean((predictions - targets) ** 2)


def optimized_mse_gradient(predictions, targets):
    """
    Optimized MSE gradient computation.
    
    Args:
        predictions: Model predictions
        targets: Target values
        
    Returns:
        Gradient w.r.t. predictions
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    
    if predictions.size > 1000:
        result = np.empty_like(predictions)
        fast_mse_gradient_impl(predictions.flatten(), targets.flatten(), result.flatten())
        return result
    else:
        return 2.0 * (predictions - targets) / predictions.size


# =============================================================================
# Memory and Data Type Optimizations
# =============================================================================

def ensure_optimal_dtype(array, target_dtype=np.float32):
    """
    Ensure array has optimal data type for performance.
    
    Args:
        array: Input array
        target_dtype: Target data type
        
    Returns:
        Array with optimal data type
    """
    if array.dtype != target_dtype:
        return array.astype(target_dtype)
    return array


def ensure_contiguous(array):
    """
    Ensure array is contiguous in memory for better cache performance.
    
    Args:
        array: Input array
        
    Returns:
        Contiguous array
    """
    if not array.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(array)
    return array


def optimize_array(array, target_dtype=np.float32):
    """
    Apply all array optimizations.
    
    Args:
        array: Input array
        target_dtype: Target data type
        
    Returns:
        Optimized array
    """
    array = ensure_optimal_dtype(array, target_dtype)
    array = ensure_contiguous(array)
    return array


# =============================================================================
# Batch Processing Optimizations
# =============================================================================

def optimal_batch_size(total_samples, memory_limit_mb=1000):
    """
    Calculate optimal batch size based on memory constraints.
    
    Args:
        total_samples: Total number of samples
        memory_limit_mb: Memory limit in MB
        
    Returns:
        Optimal batch size
    """
    # Simple heuristic based on memory and sample count
    if total_samples < 100:
        return total_samples
    elif total_samples < 1000:
        return min(32, total_samples)
    elif total_samples < 10000:
        return min(64, total_samples)
    else:
        return min(128, total_samples)


def should_use_jit_optimization(array_size, threshold=1000):
    """
    Determine if JIT optimization should be used based on array size.
    
    Args:
        array_size: Size of the array to process
        threshold: Threshold above which JIT is beneficial
        
    Returns:
        Boolean indicating whether to use JIT
    """
    return array_size > threshold


# =============================================================================
# Performance Monitoring
# =============================================================================

class PerformanceMonitor:
    """Simple performance monitoring for optimization decisions."""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
    
    def time_function(self, func_name, func, *args, **kwargs):
        """Time a function call and store the result."""
        import time
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        if func_name not in self.timings:
            self.timings[func_name] = []
            self.call_counts[func_name] = 0
        
        self.timings[func_name].append(end_time - start_time)
        self.call_counts[func_name] += 1
        
        return result
    
    def get_average_time(self, func_name):
        """Get average execution time for a function."""
        if func_name in self.timings:
            return np.mean(self.timings[func_name])
        return 0.0
    
    def should_use_optimization(self, func_name, threshold_ms=1.0):
        """Determine if optimization should be used based on timing history."""
        avg_time = self.get_average_time(func_name)
        return avg_time > threshold_ms / 1000.0  # Convert ms to seconds


# Global performance monitor
performance_monitor = PerformanceMonitor()


# =============================================================================
# Adaptive Optimization Selection
# =============================================================================

def adaptive_matrix_multiply(A, B):
    """
    Adaptively choose between NumPy and custom implementations.
    
    Args:
        A: First matrix
        B: Second matrix
        
    Returns:
        Matrix product
    """
    # For matrix multiplication, NumPy's BLAS is almost always faster
    # unless we have very specific patterns or constraints
    return np.dot(A, B)


def adaptive_element_wise_op(operation, *arrays):
    """
    Adaptively choose optimization strategy for element-wise operations.
    
    Args:
        operation: Operation name ('relu', 'sigmoid', etc.)
        *arrays: Input arrays
        
    Returns:
        Operation result
    """
    # Determine array size
    total_size = sum(arr.size for arr in arrays)
    
    # Use JIT for larger arrays, NumPy for smaller ones
    if total_size > 1000:
        if operation == 'relu_forward':
            return optimized_relu_forward(arrays[0])
        elif operation == 'relu_backward':
            return optimized_relu_backward(arrays[0], arrays[1])
        elif operation == 'sigmoid_forward':
            return optimized_sigmoid_forward(arrays[0])
        elif operation == 'sigmoid_backward':
            return optimized_sigmoid_backward(arrays[0], arrays[1])
    
    # Fallback to NumPy implementations
    if operation == 'relu_forward':
        return np.maximum(0, arrays[0])
    elif operation == 'relu_backward':
        return arrays[0] * (arrays[1] > 0)
    elif operation == 'sigmoid_forward':
        x = np.clip(arrays[0], -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    elif operation == 'sigmoid_backward':
        return arrays[0] * arrays[1] * (1.0 - arrays[1])
    
    raise ValueError(f"Unknown operation: {operation}")