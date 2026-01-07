"""
Optimized neural network implementation using JIT compilation and parallel processing.

This module provides high-performance neural network classes that leverage
all available optimizations for maximum speed while maintaining compatibility
with the original API.
"""

import numpy as np
from joblib import Parallel, delayed
from .layers import Layer
from .optimized_layers import (
    OptimizedDense, OptimizedReLU, OptimizedSigmoid, 
    OptimizedTanh, OptimizedSoftmax, BatchNormalization
)
from .optimized_losses import OptimizedMSELoss, OptimizedBCEWithLogitsLoss
from .optimized_ops import set_numba_threads, get_optimal_num_threads


class OptimizedSequential:
    """
    Optimized sequential neural network model using JIT compilation and parallel processing.
    
    This implementation provides significant performance improvements over the base Sequential
    class while maintaining full API compatibility.
    """
    
    def __init__(self, use_fast_ops=True, n_jobs=-1):
        """
        Initialize optimized sequential model.
        
        Args:
            use_fast_ops: Whether to use JIT-compiled operations (default: True)
            n_jobs: Number of parallel jobs for batch processing (-1 for all cores)
        """
        self.layers = []
        self.use_fast_ops = use_fast_ops
        self.n_jobs = n_jobs
        
        # Set optimal number of threads for Numba
        if use_fast_ops:
            set_numba_threads()
    
    def add(self, layer):
        """
        Add a layer to the network.
        
        Args:
            layer: Layer instance to add to the network
        """
        if not isinstance(layer, Layer):
            raise TypeError("Added layer must be an instance of Layer class")
        
        # Enable fast operations for optimized layers
        if hasattr(layer, 'use_fast_ops'):
            layer.use_fast_ops = self.use_fast_ops
        
        self.layers.append(layer)
    
    def forward(self, inputs):
        """
        Optimized forward propagation through all layers.
        
        Args:
            inputs: Input data to the network
            
        Returns:
            Output of the network
        """
        if not self.layers:
            raise ValueError("No layers added to the network")
        
        # Ensure inputs are float32 for optimal performance
        current_output = np.asarray(inputs, dtype=np.float32)
        
        # Pass inputs through each layer sequentially
        for layer in self.layers:
            current_output = layer.forward(current_output)
        
        return current_output
    
    def backward(self, grad_output):
        """
        Optimized backward propagation through all layers in reverse order.
        
        Args:
            grad_output: Gradient of loss with respect to network output
            
        Returns:
            Gradient of loss with respect to network input
        """
        if not self.layers:
            raise ValueError("No layers added to the network")
        
        # Ensure grad_output is float32
        current_grad = np.asarray(grad_output, dtype=np.float32)
        
        # Pass gradients through layers in reverse order
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad)
        
        return current_grad
    
    def train_step(self, inputs, targets, loss_fn, optimizer, return_regularized_loss=False):
        """
        Optimized training step with parallel batch processing support.
        
        Args:
            inputs: Input data
            targets: Target values
            loss_fn: Loss function instance
            optimizer: Optimizer instance
            return_regularized_loss: If True, return loss including regularization term
            
        Returns:
            Loss value for this step
        """
        # Forward pass
        predictions = self.forward(inputs)
        
        # Compute loss
        loss_value = loss_fn.forward(predictions, targets)
        
        # Compute loss gradient
        loss_grad = loss_fn.backward(predictions, targets)
        
        # Backward pass
        self.backward(loss_grad)
        
        # Collect all parameters and gradients from layers
        all_parameters = []
        all_gradients = []
        
        for layer in self.layers:
            layer_params = layer.get_parameters()
            layer_grads = layer.get_gradients()
            
            all_parameters.extend(layer_params)
            all_gradients.extend(layer_grads)
        
        # Update parameters using optimizer
        if all_parameters and all_gradients:
            optimizer.step(all_parameters, all_gradients)
        
        # Optionally add regularization term to reported loss
        if return_regularized_loss and hasattr(optimizer, 'weight_decay') and optimizer.weight_decay > 0:
            reg_term = 0.0
            for param in all_parameters:
                reg_term += np.sum(param ** 2)
            loss_value = loss_value + 0.5 * optimizer.weight_decay * reg_term
        
        return loss_value
    
    def predict(self, inputs):
        """
        Make predictions with optimized batch processing.
        
        Args:
            inputs: Input data
            
        Returns:
            Network predictions
        """
        return self.forward(inputs)
    
    def fit(self, X, y, epochs, batch_size, loss_fn, optimizer, 
            validation_data=None, return_regularized_loss=False, 
            verbose=True, early_stopping_patience=None):
        """
        Train the network with optimized batch processing and parallel support.
        
        Args:
            X: Training input data
            y: Training target data
            epochs: Number of training epochs
            batch_size: Size of training batches
            loss_fn: Loss function instance
            optimizer: Optimizer instance
            validation_data: Tuple of (X_val, y_val) for validation
            return_regularized_loss: If True, report loss including regularization term
            verbose: Whether to print training progress
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            Training history (losses, etc.)
        """
        # Convert inputs to numpy arrays with optimal dtype
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Initialize training history
        history = {
            'losses': [],
            'val_losses': [],
            'epochs': []
        }
        
        # Validation data
        X_val, y_val = None, None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Get number of samples
        n_samples = X.shape[0]
        
        if verbose:
            print(f"Training optimized network for {epochs} epochs")
            print(f"Training data: {X.shape}, Batch size: {batch_size}")
            print(f"Using fast operations: {self.use_fast_ops}")
            if self.use_fast_ops:
                print(f"Numba threads: {get_optimal_num_threads()}")
        
        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process batches
            if self.n_jobs == 1 or batch_size >= n_samples:
                # Sequential batch processing
                for i in range(0, n_samples, batch_size):
                    batch_end = min(i + batch_size, n_samples)
                    X_batch = X_shuffled[i:batch_end]
                    y_batch = y_shuffled[i:batch_end]
                    
                    batch_loss = self.train_step(X_batch, y_batch, loss_fn, optimizer, return_regularized_loss)
                    epoch_losses.append(batch_loss)
            else:
                # Parallel batch processing for large datasets
                batch_indices = [(i, min(i + batch_size, n_samples)) 
                               for i in range(0, n_samples, batch_size)]
                
                def process_batch(start_idx, end_idx):
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]
                    return self.train_step(X_batch, y_batch, loss_fn, optimizer, return_regularized_loss)
                
                n_jobs = self.n_jobs if self.n_jobs > 0 else -1
                batch_losses = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(process_batch)(start, end) for start, end in batch_indices
                )
                epoch_losses.extend(batch_losses)
            
            # Calculate average loss for the epoch
            avg_loss = np.mean(epoch_losses)
            history['losses'].append(avg_loss)
            history['epochs'].append(epoch)
            
            # Validation evaluation
            val_loss = None
            if X_val is not None:
                val_predictions = self.predict(X_val)
                val_loss = loss_fn.forward(val_predictions, y_val)
                history['val_losses'].append(val_loss)
                
                # Early stopping check
                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # Print progress
            if verbose:
                print_interval = max(1, min(10, epochs // 10))
                if (epoch + 1) % print_interval == 0 or epoch == 0:
                    progress_msg = f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}"
                    if val_loss is not None:
                        progress_msg += f", Val Loss: {val_loss:.6f}"
                    print(progress_msg)
        
        return history
    
    def evaluate(self, X, y, loss_fn, batch_size=1000):
        """
        Evaluate the network on test data with optimized batch processing.
        
        Args:
            X: Test input data
            y: Test target data
            loss_fn: Loss function instance
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Process in batches for memory efficiency
        n_samples = X.shape[0]
        all_predictions = []
        batch_losses = []
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            X_batch = X[i:batch_end]
            y_batch = y[i:batch_end]
            
            # Get predictions
            predictions = self.predict(X_batch)
            all_predictions.append(predictions)
            
            # Compute batch loss
            batch_loss = loss_fn.forward(predictions, y_batch)
            batch_losses.append(batch_loss * len(X_batch))  # Weight by batch size
        
        # Combine results
        all_predictions = np.concatenate(all_predictions)
        total_loss = np.sum(batch_losses) / n_samples
        
        # Compute additional metrics
        mse = np.mean((all_predictions - y) ** 2)
        mae = np.mean(np.abs(all_predictions - y))
        
        return {
            'loss': total_loss,
            'mse': mse,
            'mae': mae,
            'predictions': all_predictions
        }
    
    def get_parameters(self):
        """Get all parameters from all layers."""
        all_parameters = []
        for layer in self.layers:
            all_parameters.extend(layer.get_parameters())
        return all_parameters
    
    def get_gradients(self):
        """Get all gradients from all layers."""
        all_gradients = []
        for layer in self.layers:
            all_gradients.extend(layer.get_gradients())
        return all_gradients
    
    def summary(self):
        """Print a summary of the network architecture."""
        print("Optimized Sequential Network Summary")
        print("=" * 50)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            
            if hasattr(layer, 'weights'):
                # Dense layer
                input_size = layer.weights.shape[0]
                output_size = layer.weights.shape[1]
                layer_params = layer.weights.size + layer.biases.size
                print(f"Layer {i+1}: {layer_name} ({input_size} -> {output_size}) - {layer_params:,} params")
                total_params += layer_params
            else:
                # Activation layer
                print(f"Layer {i+1}: {layer_name} - 0 params")
        
        print("=" * 50)
        print(f"Total parameters: {total_params:,}")
        print(f"Fast operations: {self.use_fast_ops}")
        print(f"Parallel jobs: {self.n_jobs}")


class OptimizedAutoencoder:
    """
    Optimized autoencoder implementation using high-performance components.
    """
    
    def __init__(self, input_dim=784, latent_dim=32, hidden_dims=None, use_fast_ops=True):
        """
        Initialize optimized autoencoder.
        
        Args:
            input_dim: Input dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            use_fast_ops: Whether to use JIT-compiled operations
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_fast_ops = use_fast_ops
        
        if hidden_dims is None:
            self.hidden_dims = [256, 128, 64]
        else:
            self.hidden_dims = hidden_dims
        
        # Build optimized network
        self.network = self._build_network()
        self.is_trained = False
        self.training_history = None
    
    def _build_network(self):
        """Build optimized autoencoder network."""
        network = OptimizedSequential(use_fast_ops=self.use_fast_ops)
        
        current_dim = self.input_dim
        
        # Encoder layers
        for hidden_dim in self.hidden_dims:
            network.add(OptimizedDense(current_dim, hidden_dim, self.use_fast_ops))
            network.add(OptimizedReLU(self.use_fast_ops))
            current_dim = hidden_dim
        
        # Latent layer (no activation)
        network.add(OptimizedDense(current_dim, self.latent_dim, self.use_fast_ops))
        
        # Decoder layers
        current_dim = self.latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            network.add(OptimizedDense(current_dim, hidden_dim, self.use_fast_ops))
            network.add(OptimizedReLU(self.use_fast_ops))
            current_dim = hidden_dim
        
        # Output layer with sigmoid activation
        network.add(OptimizedDense(current_dim, self.input_dim, self.use_fast_ops))
        network.add(OptimizedSigmoid(self.use_fast_ops))
        
        return network
    
    def forward(self, inputs):
        """Forward pass through autoencoder."""
        return self.network.forward(inputs)
    
    def train(self, X_train, X_val=None, epochs=100, batch_size=32, 
              learning_rate=0.01, verbose=True):
        """
        Train the optimized autoencoder.
        
        Args:
            X_train: Training data
            X_val: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        from .optimizer import SGD
        
        # Create optimized loss function and optimizer
        loss_fn = OptimizedMSELoss(use_fast_ops=self.use_fast_ops)
        optimizer = SGD(learning_rate=learning_rate)
        
        # For autoencoder, input and target are the same
        validation_data = (X_val, X_val) if X_val is not None else None
        
        if verbose:
            print(f"Training optimized autoencoder (latent_dim={self.latent_dim})")
            self.network.summary()
        
        # Train the network
        history = self.network.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            loss_fn=loss_fn,
            optimizer=optimizer,
            validation_data=validation_data,
            verbose=verbose
        )
        
        self.is_trained = True
        self.training_history = history
        
        return history
    
    def encode(self, inputs):
        """Encode inputs to latent space."""
        # Forward pass through encoder layers only
        current_output = np.asarray(inputs, dtype=np.float32)
        
        # Pass through encoder layers (up to latent layer)
        encoder_layers = len(self.hidden_dims) * 2 + 1  # Dense + ReLU pairs + latent layer
        
        for i in range(encoder_layers):
            current_output = self.network.layers[i].forward(current_output)
        
        return current_output
    
    def decode(self, latent_features):
        """Decode latent features to original space."""
        # Forward pass through decoder layers only
        current_output = np.asarray(latent_features, dtype=np.float32)
        
        # Start from after the latent layer
        encoder_layers = len(self.hidden_dims) * 2 + 1
        
        for i in range(encoder_layers, len(self.network.layers)):
            current_output = self.network.layers[i].forward(current_output)
        
        return current_output
    
    def reconstruct(self, inputs):
        """Reconstruct inputs (full autoencoder pass)."""
        return self.forward(inputs)


# Factory functions for easy creation of optimized networks

def create_optimized_mlp(layer_sizes, activations=None, use_fast_ops=True):
    """
    Create an optimized multi-layer perceptron.
    
    Args:
        layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
        activations: List of activation functions for each layer (except input)
        use_fast_ops: Whether to use JIT-compiled operations
        
    Returns:
        OptimizedSequential network
    """
    if activations is None:
        activations = ['relu'] * (len(layer_sizes) - 2) + ['sigmoid']
    
    network = OptimizedSequential(use_fast_ops=use_fast_ops)
    
    for i in range(len(layer_sizes) - 1):
        # Add dense layer
        network.add(OptimizedDense(layer_sizes[i], layer_sizes[i + 1], use_fast_ops))
        
        # Add activation layer
        if i < len(activations):
            activation = activations[i].lower()
            if activation == 'relu':
                network.add(OptimizedReLU(use_fast_ops))
            elif activation == 'sigmoid':
                network.add(OptimizedSigmoid(use_fast_ops))
            elif activation == 'tanh':
                network.add(OptimizedTanh(use_fast_ops))
            elif activation == 'softmax':
                network.add(OptimizedSoftmax(use_fast_ops))
    
    return network


def create_optimized_classifier(input_size, num_classes, hidden_sizes=None, use_fast_ops=True):
    """
    Create an optimized neural network classifier.
    
    Args:
        input_size: Size of input features
        num_classes: Number of output classes
        hidden_sizes: List of hidden layer sizes
        use_fast_ops: Whether to use JIT-compiled operations
        
    Returns:
        OptimizedSequential network for classification
    """
    if hidden_sizes is None:
        hidden_sizes = [128, 64]
    
    layer_sizes = [input_size] + hidden_sizes + [num_classes]
    activations = ['relu'] * len(hidden_sizes) + ['softmax']
    
    return create_optimized_mlp(layer_sizes, activations, use_fast_ops)


def benchmark_network_performance(network_class, input_shape, num_iterations=100):
    """
    Benchmark network performance.
    
    Args:
        network_class: Network class to benchmark
        input_shape: Shape of input data
        num_iterations: Number of iterations for timing
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Create network
    if 'Autoencoder' in network_class.__name__:
        network = network_class(input_dim=input_shape[1])
    else:
        network = network_class()
        # Add some layers for testing
        network.add(OptimizedDense(input_shape[1], 64))
        network.add(OptimizedReLU())
        network.add(OptimizedDense(64, input_shape[1]))
        network.add(OptimizedSigmoid())
    
    # Generate test data
    inputs = np.random.randn(*input_shape).astype(np.float32)
    
    # Warm-up runs
    for _ in range(10):
        output = network.forward(inputs)
    
    # Time forward pass
    start_time = time.time()
    for _ in range(num_iterations):
        output = network.forward(inputs)
    forward_time = (time.time() - start_time) / num_iterations
    
    return {
        'forward_time': forward_time,
        'network_type': network_class.__name__,
        'input_shape': input_shape,
        'output_shape': output.shape
    }