"""
Neural network model implementations.

This module contains the Sequential class for building and training
neural networks by composing layers.
"""

import numpy as np
from .layers import Layer


class Sequential:
    """
    Sequential neural network model.
    
    A linear stack of layers where data flows from input to output
    through each layer in sequence.
    """
    
    def __init__(self):
        """Initialize empty sequential model."""
        self.layers = []
    
    def add(self, layer):
        """
        Add a layer to the network.
        
        Args:
            layer: Layer instance to add to the network
            
        Raises:
            TypeError: If layer is not an instance of Layer class
        """
        if not isinstance(layer, Layer):
            raise TypeError("Added layer must be an instance of Layer class")
        
        self.layers.append(layer)
    
    def forward(self, inputs):
        """
        Forward propagation through all layers.
        
        Args:
            inputs: Input data to the network
            
        Returns:
            Output of the network
            
        Raises:
            ValueError: If no layers have been added to the network
        """
        if not self.layers:
            raise ValueError("No layers added to the network")
        
        # Pass inputs through each layer sequentially
        current_output = inputs
        for layer in self.layers:
            current_output = layer.forward(current_output)
        
        return current_output
    
    def backward(self, grad_output):
        """
        Backward propagation through all layers in reverse order.
        
        Args:
            grad_output: Gradient of loss with respect to network output
            
        Returns:
            Gradient of loss with respect to network input
            
        Raises:
            ValueError: If no layers have been added to the network
        """
        if not self.layers:
            raise ValueError("No layers added to the network")
        
        # Pass gradients through layers in reverse order
        current_grad = grad_output
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad)
        
        return current_grad
    
    def train_step(self, inputs, targets, loss_fn, optimizer, return_regularized_loss=False):
        """
        Perform one training step: forward, loss, backward, optimize.
        
        Args:
            inputs: Input data
            targets: Target values
            loss_fn: Loss function instance
            optimizer: Optimizer instance
            return_regularized_loss: If True, return loss including regularization term
            
        Returns:
            Loss value for this step (with regularization if return_regularized_loss=True)
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
        Make predictions (inference mode).
        
        Args:
            inputs: Input data
            
        Returns:
            Network predictions
        """
        # Prediction is just a forward pass
        return self.forward(inputs)
    
    def fit(self, X, y, epochs, batch_size, loss_fn, optimizer, return_regularized_loss=False):
        """
        Train the network for multiple epochs.
        
        Args:
            X: Training input data
            y: Training target data
            epochs: Number of training epochs
            batch_size: Size of training batches
            loss_fn: Loss function instance
            optimizer: Optimizer instance
            return_regularized_loss: If True, report loss including regularization term
            
        Returns:
            Training history (losses, etc.)
        """
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Initialize training history
        history = {
            'losses': [],
            'epochs': []
        }
        
        # Get number of samples
        n_samples = X.shape[0]
        
        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            
            # Create batches
            for i in range(0, n_samples, batch_size):
                # Get batch
                batch_end = min(i + batch_size, n_samples)
                X_batch = X[i:batch_end]
                y_batch = y[i:batch_end]
                
                # Perform training step
                batch_loss = self.train_step(X_batch, y_batch, loss_fn, optimizer, return_regularized_loss)
                epoch_losses.append(batch_loss)
            
            # Calculate average loss for the epoch
            avg_loss = np.mean(epoch_losses)
            
            # Store history
            history['losses'].append(avg_loss)
            history['epochs'].append(epoch)
            
            # Print progress (every 10% of epochs or every 10 epochs, whichever is less frequent)
            print_interval = max(1, min(10, epochs // 10))
            if (epoch + 1) % print_interval == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return history
    
    def get_parameters(self):
        """
        Get all parameters from all layers.
        
        Returns:
            List of all parameter arrays from all layers
        """
        all_parameters = []
        for layer in self.layers:
            all_parameters.extend(layer.get_parameters())
        return all_parameters
    
    def get_gradients(self):
        """
        Get all gradients from all layers.
        
        Returns:
            List of all gradient arrays from all layers
        """
        all_gradients = []
        for layer in self.layers:
            all_gradients.extend(layer.get_gradients())
        return all_gradients