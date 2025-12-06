"""
Gradient checking utilities for validating backpropagation correctness.

This module provides tools to numerically verify that analytical gradients
computed during backpropagation are correct by comparing them with numerical
gradients computed using finite differences.
"""

import numpy as np
from .layers import Layer
from .losses import MSELoss


class GradientChecker:
    """
    Gradient checking utility for validating backpropagation implementation.
    
    This class provides methods to compute numerical gradients using finite differences
    and compare them with analytical gradients to ensure backpropagation correctness.
    """
    
    def __init__(self, epsilon=1e-7, tolerance=1e-5):
        """
        Initialize gradient checker.
        
        Args:
            epsilon (float): Small value for finite difference computation
            tolerance (float): Relative error threshold for gradient comparison
        """
        self.epsilon = epsilon
        self.tolerance = tolerance
    
    def compute_numerical_gradient(self, layer, inputs, grad_output):
        """
        Compute numerical gradients for a layer's parameters using finite differences.
        
        Uses the formula: ∂L/∂W ≈ [L(W + ε) - L(W - ε)] / (2ε)
        
        Args:
            layer: Layer instance to check
            inputs: Input data to the layer
            grad_output: Gradient of loss w.r.t. layer output
            
        Returns:
            List of numerical gradients for each parameter
        """
        parameters = layer.get_parameters()
        numerical_gradients = []
        
        for param in parameters:
            # Initialize numerical gradient array with same shape as parameter
            numerical_grad = np.zeros_like(param)
            
            # Iterate through each element of the parameter
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                
                # Store original value
                original_value = param[idx]
                
                # Compute f(x + epsilon)
                param[idx] = original_value + self.epsilon
                output_plus = layer.forward(inputs)
                loss_plus = np.sum(grad_output * output_plus)
                
                # Compute f(x - epsilon)
                param[idx] = original_value - self.epsilon
                output_minus = layer.forward(inputs)
                loss_minus = np.sum(grad_output * output_minus)
                
                # Compute numerical gradient
                numerical_grad[idx] = (loss_plus - loss_minus) / (2 * self.epsilon)
                
                # Restore original value
                param[idx] = original_value
                
                it.iternext()
            
            numerical_gradients.append(numerical_grad)
        
        return numerical_gradients
    
    def compute_relative_error(self, analytical_grad, numerical_grad):
        """
        Compute relative error between analytical and numerical gradients.
        
        Uses the formula: |analytical - numerical| / max(|analytical|, |numerical|)
        
        Args:
            analytical_grad: Analytically computed gradient
            numerical_grad: Numerically computed gradient
            
        Returns:
            Relative error as a scalar value
        """
        # Flatten gradients for easier computation
        analytical_flat = analytical_grad.flatten()
        numerical_flat = numerical_grad.flatten()
        
        # Compute absolute differences
        diff = np.abs(analytical_flat - numerical_flat)
        
        # Compute denominators (max of absolute values)
        denominators = np.maximum(np.abs(analytical_flat), np.abs(numerical_flat))
        
        # Avoid division by zero by adding small epsilon where both gradients are zero
        denominators = np.maximum(denominators, 1e-12)
        
        # Compute relative errors
        relative_errors = diff / denominators
        
        # Return maximum relative error
        return np.max(relative_errors)
    
    def check_layer_gradients(self, layer, inputs, grad_output):
        """
        Check gradients for a single layer.
        
        Args:
            layer: Layer instance to check
            inputs: Input data to the layer
            grad_output: Gradient of loss w.r.t. layer output
            
        Returns:
            Dictionary containing check results
        """
        # Perform forward pass to initialize layer state
        layer.forward(inputs)
        
        # Perform backward pass to compute analytical gradients
        layer.backward(grad_output)
        
        # Get analytical gradients
        analytical_gradients = layer.get_gradients()
        
        # Compute numerical gradients
        numerical_gradients = self.compute_numerical_gradient(layer, inputs, grad_output)
        
        # Compare gradients
        results = {
            'layer_type': type(layer).__name__,
            'passed': True,
            'parameter_results': []
        }
        
        parameter_names = ['weights', 'biases'] if hasattr(layer, 'weights') else []
        
        for i, (analytical_grad, numerical_grad) in enumerate(zip(analytical_gradients, numerical_gradients)):
            if analytical_grad is None:
                continue
                
            # Compute relative error
            relative_error = self.compute_relative_error(analytical_grad, numerical_grad)
            
            # Check if gradient passes tolerance test
            param_passed = relative_error < self.tolerance
            
            param_name = parameter_names[i] if i < len(parameter_names) else f'param_{i}'
            
            param_result = {
                'parameter': param_name,
                'relative_error': relative_error,
                'passed': param_passed,
                'analytical_grad_norm': np.linalg.norm(analytical_grad),
                'numerical_grad_norm': np.linalg.norm(numerical_grad)
            }
            
            results['parameter_results'].append(param_result)
            
            # Update overall pass status
            if not param_passed:
                results['passed'] = False
        
        return results
    
    def check_network_gradients(self, network, inputs, targets, loss_fn=None):
        """
        Check gradients for an entire network.
        
        Args:
            network: Sequential network instance
            inputs: Input data
            targets: Target data
            loss_fn: Loss function (defaults to MSELoss)
            
        Returns:
            Dictionary containing check results for all layers
        """
        if loss_fn is None:
            loss_fn = MSELoss()
        
        # Perform forward pass
        predictions = network.forward(inputs)
        
        # Compute loss gradient
        loss_grad = loss_fn.backward(predictions, targets)
        
        # Check gradients for each layer with parameters
        results = {
            'network_passed': True,
            'layer_results': []
        }
        
        # We need to propagate gradients backward through the network
        # to get the correct grad_output for each layer
        current_grad = loss_grad
        
        # Check layers in reverse order (as gradients flow backward)
        for i, layer in enumerate(reversed(network.layers)):
            if layer.get_parameters():  # Only check layers with parameters
                # Get the inputs to this layer by doing a partial forward pass
                layer_inputs = inputs
                for forward_layer in network.layers[:len(network.layers)-1-i]:
                    layer_inputs = forward_layer.forward(layer_inputs)
                
                # Check this layer's gradients
                layer_result = self.check_layer_gradients(layer, layer_inputs, current_grad)
                layer_result['layer_index'] = len(network.layers) - 1 - i
                results['layer_results'].append(layer_result)
                
                if not layer_result['passed']:
                    results['network_passed'] = False
            
            # Propagate gradient backward for next layer
            current_grad = layer.backward(current_grad)
        
        # Reverse layer results to match forward order
        results['layer_results'] = list(reversed(results['layer_results']))
        
        return results
    
    def print_results(self, results):
        """
        Print gradient checking results in a readable format.
        
        Args:
            results: Results dictionary from check_layer_gradients or check_network_gradients
        """
        if 'layer_results' in results:
            # Network results
            print(f"=== Network Gradient Check Results ===")
            print(f"Overall Status: {'PASSED' if results['network_passed'] else 'FAILED'}")
            print()
            
            for layer_result in results['layer_results']:
                self._print_layer_result(layer_result)
                print()
        else:
            # Single layer results
            self._print_layer_result(results)
    
    def _print_layer_result(self, layer_result):
        """Print results for a single layer."""
        layer_name = layer_result['layer_type']
        layer_index = layer_result.get('layer_index', '')
        index_str = f" (Layer {layer_index})" if layer_index != '' else ""
        
        print(f"{layer_name}{index_str}: {'PASSED' if layer_result['passed'] else 'FAILED'}")
        
        for param_result in layer_result['parameter_results']:
            status = "PASSED" if param_result['passed'] else "FAILED"
            print(f"  {param_result['parameter']}: {status}")
            print(f"    Relative Error: {param_result['relative_error']:.2e}")
            print(f"    Tolerance: {self.tolerance:.2e}")
            print(f"    Analytical Grad Norm: {param_result['analytical_grad_norm']:.6f}")
            print(f"    Numerical Grad Norm: {param_result['numerical_grad_norm']:.6f}")


def check_gradients(layer_or_network, inputs, targets_or_grad_output=None, 
                   loss_fn=None, epsilon=1e-7, tolerance=1e-5, verbose=True):
    """
    Convenience function for gradient checking.
    
    Args:
        layer_or_network: Layer or Sequential network to check
        inputs: Input data
        targets_or_grad_output: Target data (for networks) or grad_output (for layers)
        loss_fn: Loss function (for networks, defaults to MSELoss)
        epsilon: Small value for finite difference computation
        tolerance: Relative error threshold
        verbose: Whether to print results
        
    Returns:
        Dictionary containing check results
    """
    checker = GradientChecker(epsilon=epsilon, tolerance=tolerance)
    
    # Determine if we're checking a layer or network
    if isinstance(layer_or_network, Layer):
        # Single layer check
        if targets_or_grad_output is None:
            raise ValueError("grad_output must be provided for layer gradient checking")
        results = checker.check_layer_gradients(layer_or_network, inputs, targets_or_grad_output)
    else:
        # Network check
        if targets_or_grad_output is None:
            raise ValueError("targets must be provided for network gradient checking")
        results = checker.check_network_gradients(layer_or_network, inputs, targets_or_grad_output, loss_fn)
    
    if verbose:
        checker.print_results(results)
    
    return results