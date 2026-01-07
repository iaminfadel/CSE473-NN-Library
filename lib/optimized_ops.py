"""
Optimized operations using Numba JIT compilation for performance-critical functions.

This module provides JIT-compiled versions of core mathematical operations
used throughout the neural network library for significant performance improvements.
"""

import numpy as np
from numba import jit, prange
from typing import Tuple


# =============================================================================
# Matrix Operations
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def fast_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Fast matrix multiplication using Numba JIT compilation.
    
    Note: For small matrices, NumPy's optimized BLAS is often faster.
    This function is most beneficial for medium-sized matrices where
    the JIT compilation overhead is amortized.
    
    Args:
        A: Matrix A of shape (m, k)
        B: Matrix B of shape (k, n)
        
    Returns:
        Result matrix of shape (m, n)
    """
    # For small matrices, fall back to a simple implementation
    # that doesn't use parallel loops to avoid overhead
    return np.dot(A, B)


@jit(nopython=True, cache=True, fastmath=True)
def fast_add_bias(inputs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Fast bias addition with broadcasting.
    
    Args:
        inputs: Input matrix of shape (batch_size, features)
        bias: Bias vector of shape (1, features) or (features,)
        
    Returns:
        Result with bias added
    """
    # Simple broadcasting is often faster than manual loops for this operation
    return inputs + bias


@jit(nopython=True, cache=True, fastmath=True)
def fast_gradient_weights(inputs: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    """
    Fast computation of weight gradients: inputs.T @ grad_output
    
    Args:
        inputs: Input matrix of shape (batch_size, input_features)
        grad_output: Gradient matrix of shape (batch_size, output_features)
        
    Returns:
        Weight gradients of shape (input_features, output_features)
    """
    # Use NumPy's optimized matrix multiplication
    return np.dot(inputs.T, grad_output)


@jit(nopython=True, cache=True, fastmath=True)
def fast_gradient_bias(grad_output: np.ndarray) -> np.ndarray:
    """
    Fast computation of bias gradients: sum over batch dimension
    
    Args:
        grad_output: Gradient matrix of shape (batch_size, features)
        
    Returns:
        Bias gradients of shape (1, features)
    """
    # Use NumPy's optimized sum with keepdims
    return np.sum(grad_output, axis=0, keepdims=True)


# =============================================================================
# Activation Functions
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def fast_relu_forward(x: np.ndarray) -> np.ndarray:
    """
    Fast ReLU forward pass: max(0, x)
    
    Args:
        x: Input array
        
    Returns:
        ReLU output
    """
    result = np.empty_like(x)
    flat_x = x.flatten()
    flat_result = result.flatten()
    
    for i in prange(flat_x.size):
        flat_result[i] = max(0.0, flat_x[i])
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def fast_relu_backward(grad_output: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """
    Fast ReLU backward pass: grad_output * (inputs > 0)
    
    Args:
        grad_output: Gradient from next layer
        inputs: Original inputs to ReLU
        
    Returns:
        Gradient w.r.t. inputs
    """
    result = np.empty_like(grad_output)
    flat_grad = grad_output.flatten()
    flat_inputs = inputs.flatten()
    flat_result = result.flatten()
    
    for i in prange(flat_grad.size):
        if flat_inputs[i] > 0.0:
            flat_result[i] = flat_grad[i]
        else:
            flat_result[i] = 0.0
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def fast_sigmoid_forward(x: np.ndarray) -> np.ndarray:
    """
    Fast sigmoid forward pass with numerical stability.
    
    Args:
        x: Input array
        
    Returns:
        Sigmoid output
    """
    result = np.empty_like(x)
    flat_x = x.flatten()
    flat_result = result.flatten()
    
    for i in prange(flat_x.size):
        val = flat_x[i]
        # Clip for numerical stability
        if val > 500.0:
            val = 500.0
        elif val < -500.0:
            val = -500.0
        
        if val >= 0:
            exp_neg = np.exp(-val)
            flat_result[i] = 1.0 / (1.0 + exp_neg)
        else:
            exp_pos = np.exp(val)
            flat_result[i] = exp_pos / (1.0 + exp_pos)
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def fast_sigmoid_backward(grad_output: np.ndarray, sigmoid_output: np.ndarray) -> np.ndarray:
    """
    Fast sigmoid backward pass: grad_output * sigmoid_output * (1 - sigmoid_output)
    
    Args:
        grad_output: Gradient from next layer
        sigmoid_output: Output from sigmoid forward pass
        
    Returns:
        Gradient w.r.t. inputs
    """
    result = np.empty_like(grad_output)
    flat_grad = grad_output.flatten()
    flat_sigmoid = sigmoid_output.flatten()
    flat_result = result.flatten()
    
    for i in prange(flat_grad.size):
        s = flat_sigmoid[i]
        flat_result[i] = flat_grad[i] * s * (1.0 - s)
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def fast_tanh_forward(x: np.ndarray) -> np.ndarray:
    """
    Fast tanh forward pass.
    
    Args:
        x: Input array
        
    Returns:
        Tanh output
    """
    result = np.empty_like(x)
    flat_x = x.flatten()
    flat_result = result.flatten()
    
    for i in prange(flat_x.size):
        flat_result[i] = np.tanh(flat_x[i])
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def fast_tanh_backward(grad_output: np.ndarray, tanh_output: np.ndarray) -> np.ndarray:
    """
    Fast tanh backward pass: grad_output * (1 - tanh_output^2)
    
    Args:
        grad_output: Gradient from next layer
        tanh_output: Output from tanh forward pass
        
    Returns:
        Gradient w.r.t. inputs
    """
    result = np.empty_like(grad_output)
    flat_grad = grad_output.flatten()
    flat_tanh = tanh_output.flatten()
    flat_result = result.flatten()
    
    for i in prange(flat_grad.size):
        t = flat_tanh[i]
        flat_result[i] = flat_grad[i] * (1.0 - t * t)
    
    return result


# =============================================================================
# Loss Functions
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def fast_mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Fast MSE loss computation.
    
    Args:
        predictions: Model predictions
        targets: Target values
        
    Returns:
        MSE loss value
    """
    flat_pred = predictions.flatten()
    flat_targets = targets.flatten()
    
    total_loss = 0.0
    n_elements = flat_pred.size
    
    for i in prange(n_elements):
        diff = flat_pred[i] - flat_targets[i]
        total_loss += diff * diff
    
    return total_loss / n_elements


@jit(nopython=True, parallel=True, cache=True)
def fast_mse_gradient(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Fast MSE gradient computation.
    
    Args:
        predictions: Model predictions
        targets: Target values
        
    Returns:
        Gradient w.r.t. predictions
    """
    result = np.empty_like(predictions)
    flat_pred = predictions.flatten()
    flat_targets = targets.flatten()
    flat_result = result.flatten()
    
    n_elements = flat_pred.size
    scale = 2.0 / n_elements
    
    for i in prange(n_elements):
        flat_result[i] = scale * (flat_pred[i] - flat_targets[i])
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def fast_bce_with_logits_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Fast BCE with logits loss computation (numerically stable).
    
    Args:
        logits: Raw model outputs (before sigmoid)
        targets: Target values (0 or 1)
        
    Returns:
        BCE loss value
    """
    flat_logits = logits.flatten()
    flat_targets = targets.flatten()
    
    total_loss = 0.0
    n_elements = flat_logits.size
    
    for i in prange(n_elements):
        x = flat_logits[i]
        z = flat_targets[i]
        
        # Numerically stable BCE with logits: max(x, 0) - x * z + log(1 + exp(-abs(x)))
        max_val = max(x, 0.0)
        total_loss += max_val - x * z + np.log(1.0 + np.exp(-abs(x)))
    
    return total_loss / n_elements


@jit(nopython=True, parallel=True, cache=True)
def fast_bce_with_logits_gradient(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Fast BCE with logits gradient computation.
    
    Args:
        logits: Raw model outputs
        targets: Target values
        
    Returns:
        Gradient w.r.t. logits
    """
    result = np.empty_like(logits)
    flat_logits = logits.flatten()
    flat_targets = targets.flatten()
    flat_result = result.flatten()
    
    n_elements = flat_logits.size
    
    for i in prange(n_elements):
        x = flat_logits[i]
        z = flat_targets[i]
        
        # Compute sigmoid in numerically stable way
        if x >= 0:
            sigmoid_x = 1.0 / (1.0 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            sigmoid_x = exp_x / (1.0 + exp_x)
        
        # Gradient: (sigmoid(x) - z) / n_elements
        flat_result[i] = (sigmoid_x - z) / n_elements
    
    return result


# =============================================================================
# SVM Kernel Functions
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def fast_linear_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Fast linear kernel computation: X1 @ X2.T
    
    Args:
        X1: First set of vectors (n1, features)
        X2: Second set of vectors (n2, features)
        
    Returns:
        Kernel matrix (n1, n2)
    """
    n1, features = X1.shape
    n2, _ = X2.shape
    
    K = np.zeros((n1, n2), dtype=X1.dtype)
    
    for i in prange(n1):
        for j in prange(n2):
            dot_product = 0.0
            for k in range(features):
                dot_product += X1[i, k] * X2[j, k]
            K[i, j] = dot_product
    
    return K


@jit(nopython=True, parallel=True, cache=True)
def fast_rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """
    Fast RBF kernel computation: exp(-gamma * ||x1 - x2||^2)
    
    Args:
        X1: First set of vectors (n1, features)
        X2: Second set of vectors (n2, features)
        gamma: RBF kernel parameter
        
    Returns:
        Kernel matrix (n1, n2)
    """
    n1, features = X1.shape
    n2, _ = X2.shape
    
    K = np.zeros((n1, n2), dtype=X1.dtype)
    
    for i in prange(n1):
        for j in prange(n2):
            squared_distance = 0.0
            for k in range(features):
                diff = X1[i, k] - X2[j, k]
                squared_distance += diff * diff
            K[i, j] = np.exp(-gamma * squared_distance)
    
    return K


@jit(nopython=True, cache=True)
def fast_kernel_value(x1: np.ndarray, x2: np.ndarray, kernel_type: int, gamma: float) -> float:
    """
    Fast single kernel value computation.
    
    Args:
        x1: First vector
        x2: Second vector
        kernel_type: 0 for linear, 1 for RBF
        gamma: RBF parameter (ignored for linear)
        
    Returns:
        Kernel value
    """
    if kernel_type == 0:  # Linear kernel
        dot_product = 0.0
        for i in range(x1.size):
            dot_product += x1[i] * x2[i]
        return dot_product
    else:  # RBF kernel
        squared_distance = 0.0
        for i in range(x1.size):
            diff = x1[i] - x2[i]
            squared_distance += diff * diff
        return np.exp(-gamma * squared_distance)


# =============================================================================
# Batch Processing Utilities
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def fast_batch_norm(X: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast batch normalization computation.
    
    Args:
        X: Input data
        axis: Axis along which to normalize
        
    Returns:
        Tuple of (normalized_X, mean, std)
    """
    if axis == 0:
        # Normalize along batch dimension
        batch_size, features = X.shape
        mean = np.zeros(features, dtype=X.dtype)
        
        # Compute mean
        for j in prange(features):
            for i in range(batch_size):
                mean[j] += X[i, j]
            mean[j] /= batch_size
        
        # Compute variance
        var = np.zeros(features, dtype=X.dtype)
        for j in prange(features):
            for i in range(batch_size):
                diff = X[i, j] - mean[j]
                var[j] += diff * diff
            var[j] /= batch_size
        
        # Compute standard deviation
        std = np.sqrt(var + 1e-8)  # Add small epsilon for numerical stability
        
        # Normalize
        normalized = np.empty_like(X)
        for i in prange(batch_size):
            for j in prange(features):
                normalized[i, j] = (X[i, j] - mean[j]) / std[j]
        
        return normalized, mean, std
    else:
        raise NotImplementedError("Only axis=0 is currently supported")


@jit(nopython=True, parallel=True, cache=True)
def fast_shuffle_indices(n: int, seed: int = 42) -> np.ndarray:
    """
    Fast generation of shuffled indices.
    
    Args:
        n: Number of indices
        seed: Random seed
        
    Returns:
        Shuffled indices array
    """
    np.random.seed(seed)
    indices = np.arange(n)
    
    # Fisher-Yates shuffle
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    
    return indices


# =============================================================================
# Memory-Efficient Operations
# =============================================================================

@jit(nopython=True, cache=True)
def fast_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Fast dot product computation.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Dot product
    """
    result = 0.0
    for i in range(a.size):
        result += a[i] * b[i]
    
    return result


@jit(nopython=True, cache=True)
def fast_vector_norm(x: np.ndarray) -> float:
    """
    Fast L2 norm computation.
    
    Args:
        x: Input vector
        
    Returns:
        L2 norm
    """
    norm_squared = 0.0
    for i in range(x.size):
        norm_squared += x[i] * x[i]
    
    return np.sqrt(norm_squared)


@jit(nopython=True, parallel=True, cache=True)
def fast_element_wise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Fast element-wise multiplication.
    
    Args:
        a: First array
        b: Second array
        
    Returns:
        Element-wise product
    """
    result = np.empty_like(a)
    flat_a = a.flatten()
    flat_b = b.flatten()
    flat_result = result.flatten()
    
    for i in prange(flat_a.size):
        flat_result[i] = flat_a[i] * flat_b[i]
    
    return result


# =============================================================================
# Utility Functions
# =============================================================================

def get_optimal_num_threads() -> int:
    """
    Get optimal number of threads for parallel operations.
    
    Returns:
        Number of threads to use
    """
    import os
    
    # Check environment variables
    if 'NUMBA_NUM_THREADS' in os.environ:
        return int(os.environ['NUMBA_NUM_THREADS'])
    
    # Use number of CPU cores
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except:
        return 4  # Default fallback


def set_numba_threads(num_threads: int = None):
    """
    Set number of threads for Numba operations.
    
    Args:
        num_threads: Number of threads (None for auto-detect)
    """
    import os
    from numba import set_num_threads
    
    if num_threads is None:
        num_threads = get_optimal_num_threads()
    
    set_num_threads(num_threads)
    os.environ['NUMBA_NUM_THREADS'] = str(num_threads)
    
    print(f"Set Numba to use {num_threads} threads")


# Initialize optimal threading on import
set_numba_threads()