"""
Smart optimizations that provide real performance improvements.

This module implements optimizations that are carefully chosen based on
actual performance characteristics rather than theoretical benefits.
"""

import numpy as np
from numba import jit
import warnings

# Suppress Numba warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numba')


# =============================================================================
# Smart Optimization Decisions
# =============================================================================

def should_use_jit(array_size, operation_complexity=1):
    """
    Determine if JIT compilation should be used based on array size and operation complexity.
    
    Args:
        array_size: Total number of elements to process
        operation_complexity: Relative complexity of the operation (1=simple, 2=medium, 3=complex)
        
    Returns:
        Boolean indicating whether JIT should be used
    """
    # JIT compilation has overhead, so only use it for larger operations
    threshold = 10000 / operation_complexity
    return array_size > threshold


# =============================================================================
# Optimized Element-wise Operations (where JIT excels)
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def jit_relu_forward(x_flat, result_flat):
    """JIT-compiled ReLU forward pass for large arrays."""
    for i in range(x_flat.size):
        result_flat[i] = max(0.0, x_flat[i])


@jit(nopython=True, cache=True, fastmath=True)
def jit_relu_backward(grad_flat, input_flat, result_flat):
    """JIT-compiled ReLU backward pass for large arrays."""
    for i in range(grad_flat.size):
        if input_flat[i] > 0.0:
            result_flat[i] = grad_flat[i]
        else:
            result_flat[i] = 0.0


@jit(nopython=True, cache=True, fastmath=True)
def jit_sigmoid_forward(x_flat, result_flat):
    """JIT-compiled sigmoid forward pass for large arrays."""
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


# =============================================================================
# Smart Layer Implementations
# =============================================================================

class SmartDense:
    """
    Smart dense layer that chooses optimal implementation based on operation size.
    """
    
    def __init__(self, input_size, output_size):
        """Initialize smart dense layer."""
        self.input_size = input_size
        self.output_size = output_size
        
        # Use float32 for better performance
        xavier_std = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size).astype(np.float32) * xavier_std
        self.biases = np.zeros((1, output_size), dtype=np.float32)
        
        # Ensure contiguous memory layout
        self.weights = np.ascontiguousarray(self.weights)
        self.biases = np.ascontiguousarray(self.biases)
        
        self.grad_weights = None
        self.grad_biases = None
        self.last_inputs = None
    
    def forward(self, inputs):
        """Smart forward pass."""
        # Ensure optimal data type and layout
        inputs = np.asarray(inputs, dtype=np.float32)
        if not inputs.flags['C_CONTIGUOUS']:
            inputs = np.ascontiguousarray(inputs)
        
        self.last_inputs = inputs.copy()
        
        # Use NumPy's optimized BLAS for matrix multiplication
        # (Almost always faster than custom implementations)
        output = np.dot(inputs, self.weights) + self.biases
        
        return output
    
    def backward(self, grad_output):
        """Smart backward pass."""
        grad_output = np.asarray(grad_output, dtype=np.float32)
        if not grad_output.flags['C_CONTIGUOUS']:
            grad_output = np.ascontiguousarray(grad_output)
        
        # Use NumPy's optimized operations
        self.grad_weights = np.dot(self.last_inputs.T, grad_output)
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input
    
    def get_parameters(self):
        return [self.weights, self.biases]
    
    def get_gradients(self):
        return [self.grad_weights, self.grad_biases]


class SmartReLU:
    """
    Smart ReLU that chooses between NumPy and JIT based on array size.
    """
    
    def __init__(self):
        self.last_inputs = None
    
    def forward(self, inputs):
        """Smart ReLU forward pass."""
        inputs = np.asarray(inputs, dtype=np.float32)
        self.last_inputs = inputs.copy()
        
        # Choose implementation based on array size
        if should_use_jit(inputs.size, operation_complexity=1):
            # Use JIT for large arrays
            result = np.empty_like(inputs)
            jit_relu_forward(inputs.flatten(), result.flatten())
            return result
        else:
            # Use NumPy for small arrays
            return np.maximum(0, inputs)
    
    def backward(self, grad_output):
        """Smart ReLU backward pass."""
        grad_output = np.asarray(grad_output, dtype=np.float32)
        
        if should_use_jit(grad_output.size, operation_complexity=1):
            # Use JIT for large arrays
            result = np.empty_like(grad_output)
            jit_relu_backward(grad_output.flatten(), self.last_inputs.flatten(), result.flatten())
            return result
        else:
            # Use NumPy for small arrays
            return grad_output * (self.last_inputs > 0)


class SmartSigmoid:
    """
    Smart Sigmoid that chooses optimal implementation.
    """
    
    def __init__(self):
        self.last_output = None
    
    def forward(self, inputs):
        """Smart sigmoid forward pass."""
        inputs = np.asarray(inputs, dtype=np.float32)
        
        if should_use_jit(inputs.size, operation_complexity=2):
            # Use JIT for large arrays
            result = np.empty_like(inputs)
            jit_sigmoid_forward(inputs.flatten(), result.flatten())
            self.last_output = result.copy()
            return result
        else:
            # Use NumPy for small arrays
            clipped = np.clip(inputs, -500, 500)
            result = 1.0 / (1.0 + np.exp(-clipped))
            self.last_output = result.copy()
            return result
    
    def backward(self, grad_output):
        """Smart sigmoid backward pass."""
        grad_output = np.asarray(grad_output, dtype=np.float32)
        
        # Sigmoid derivative is simple enough that NumPy is usually fine
        return grad_output * self.last_output * (1.0 - self.last_output)


class SmartMSELoss:
    """
    Smart MSE loss that optimizes based on data size.
    """
    
    def forward(self, predictions, targets):
        """Smart MSE forward pass."""
        predictions = np.asarray(predictions, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        
        # MSE is simple enough that NumPy is usually optimal
        return np.mean((predictions - targets) ** 2)
    
    def backward(self, predictions, targets):
        """Smart MSE backward pass."""
        predictions = np.asarray(predictions, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        
        return 2.0 * (predictions - targets) / predictions.size


# =============================================================================
# Performance Monitoring and Adaptive Optimization
# =============================================================================

class PerformanceTracker:
    """Track performance of different implementations to make smart choices."""
    
    def __init__(self):
        self.timings = {}
        self.size_thresholds = {}
    
    def time_operation(self, operation_name, size, implementation, func, *args, **kwargs):
        """Time an operation and update thresholds."""
        import time
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        key = f"{operation_name}_{implementation}"
        if key not in self.timings:
            self.timings[key] = []
        
        self.timings[key].append((size, end_time - start_time))
        
        return result
    
    def get_optimal_threshold(self, operation_name):
        """Get the optimal size threshold for switching implementations."""
        numpy_key = f"{operation_name}_numpy"
        jit_key = f"{operation_name}_jit"
        
        if numpy_key not in self.timings or jit_key not in self.timings:
            return 10000  # Default threshold
        
        # Find crossover point where JIT becomes faster
        # This is a simplified heuristic
        numpy_times = self.timings[numpy_key]
        jit_times = self.timings[jit_key]
        
        # For now, return a reasonable default
        # In a full implementation, we'd analyze the timing data
        return 10000


# Global performance tracker
performance_tracker = PerformanceTracker()


# =============================================================================
# Benchmarking Functions
# =============================================================================

def benchmark_smart_vs_original(component_type, input_shape, iterations=100):
    """
    Benchmark smart implementations against original ones.
    
    Args:
        component_type: Type of component ('dense', 'relu', 'sigmoid', 'mse')
        input_shape: Shape of input data
        iterations: Number of iterations
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    # Generate test data
    if component_type in ['dense']:
        inputs = np.random.randn(*input_shape).astype(np.float32)
    else:
        inputs = np.random.randn(*input_shape).astype(np.float32)
    
    # Import original components
    if component_type == 'dense':
        from .layers import Dense
        original = Dense(input_shape[1], input_shape[1])
        smart = SmartDense(input_shape[1], input_shape[1])
        
        # Set same weights for fair comparison
        smart.weights = original.weights.astype(np.float32)
        smart.biases = original.biases.astype(np.float32)
        
    elif component_type == 'relu':
        from .activations import ReLU
        original = ReLU()
        smart = SmartReLU()
        
    elif component_type == 'sigmoid':
        from .activations import Sigmoid
        original = Sigmoid()
        smart = SmartSigmoid()
        
    elif component_type == 'mse':
        from .losses import MSELoss
        original = MSELoss()
        smart = SmartMSELoss()
        
        # For loss functions, we need predictions and targets
        targets = np.random.randn(*input_shape).astype(np.float32)
        
        # Warm-up
        for _ in range(5):
            original.forward(inputs, targets)
            smart.forward(inputs, targets)
        
        # Time original
        start_time = time.time()
        for _ in range(iterations):
            loss = original.forward(inputs, targets)
            grad = original.backward(inputs, targets)
        original_time = time.time() - start_time
        
        # Time smart
        start_time = time.time()
        for _ in range(iterations):
            loss = smart.forward(inputs, targets)
            grad = smart.backward(inputs, targets)
        smart_time = time.time() - start_time
        
        speedup = original_time / smart_time
        
        return {
            'component_type': component_type,
            'input_shape': input_shape,
            'original_time': original_time,
            'smart_time': smart_time,
            'speedup': speedup,
            'iterations': iterations
        }
    
    else:
        raise ValueError(f"Unsupported component type: {component_type}")
    
    # For layers and activations
    # Warm-up
    for _ in range(5):
        output_orig = original.forward(inputs)
        output_smart = smart.forward(inputs)
        
        grad_output = np.random.randn(*output_orig.shape).astype(np.float32)
        original.backward(grad_output)
        smart.backward(grad_output)
    
    # Time original
    start_time = time.time()
    for _ in range(iterations):
        output = original.forward(inputs)
        grad_output = np.random.randn(*output.shape).astype(np.float32)
        original.backward(grad_output)
    original_time = time.time() - start_time
    
    # Time smart
    start_time = time.time()
    for _ in range(iterations):
        output = smart.forward(inputs)
        grad_output = np.random.randn(*output.shape).astype(np.float32)
        smart.backward(grad_output)
    smart_time = time.time() - start_time
    
    speedup = original_time / smart_time
    
    return {
        'component_type': component_type,
        'input_shape': input_shape,
        'original_time': original_time,
        'smart_time': smart_time,
        'speedup': speedup,
        'iterations': iterations
    }


def run_comprehensive_smart_benchmark():
    """Run comprehensive benchmarks of smart optimizations."""
    print("Smart Optimization Benchmark Suite")
    print("=" * 50)
    
    # Test different sizes and components
    test_configs = [
        ('dense', (32, 100)),
        ('dense', (100, 784)),
        ('relu', (1000, 512)),
        ('relu', (100, 100)),
        ('sigmoid', (1000, 256)),
        ('sigmoid', (50, 50)),
        ('mse', (100, 10)),
        ('mse', (1000, 100))
    ]
    
    results = []
    
    for component_type, input_shape in test_configs:
        print(f"\nTesting {component_type} with shape {input_shape}...")
        
        try:
            result = benchmark_smart_vs_original(component_type, input_shape, iterations=100)
            results.append(result)
            
            print(f"  Original: {result['original_time']:.4f}s")
            print(f"  Smart:    {result['smart_time']:.4f}s")
            print(f"  Speedup:  {result['speedup']:.2f}x")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Calculate overall performance
    if results:
        speedups = [r['speedup'] for r in results]
        geometric_mean = np.prod(speedups) ** (1.0 / len(speedups))
        
        print(f"\n" + "=" * 50)
        print("OVERALL RESULTS")
        print("=" * 50)
        print(f"Geometric mean speedup: {geometric_mean:.2f}x")
        
        if geometric_mean > 1.2:
            print("🚀 Smart optimizations are working well!")
        elif geometric_mean > 1.0:
            print("⚡ Smart optimizations provide modest improvements")
        else:
            print("⚠️  Smart optimizations need tuning")
    
    return results