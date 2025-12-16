"""
Autoencoder architecture implementations with fixed sigmoid saturation issue.

This module provides modular encoder and decoder networks for building
autoencoders with configurable latent dimensions and sharp reconstructions.

Key fix: Proper weight initialization in the final decoder layer to prevent sigmoid saturation.
"""

import numpy as np
from .network import Sequential
from .layers import Dense
from .activations import ReLU, Sigmoid
from .losses import MSELoss, BCEWithLogitsLoss
from .optimizer import SGD


class Encoder:
    """
    Encoder network for autoencoder architecture.
    
    Compresses input data from 784 dimensions (MNIST flattened) to a configurable
    latent dimension through a series of Dense layers with ReLU activations.
    
    Default architecture: 784 → 256 → 128 → 64 → 32
    """
    
    def __init__(self, input_dim=784, latent_dim=32, hidden_dims=None):
        """
        Initialize encoder network.
        
        Args:
            input_dim (int): Input dimension (default: 784 for MNIST)
            latent_dim (int): Latent space dimension (default: 32)
            hidden_dims (list, optional): Hidden layer dimensions. 
                                        If None, uses [256, 128, 64]
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Default hidden dimensions if not specified
        if hidden_dims is None:
            self.hidden_dims = [256, 128, 64]
        else:
            self.hidden_dims = hidden_dims
            
        self.network = None
        self._build_network()
    
    def _build_network(self):
        """Build encoder with proper initialization."""
        self.network = Sequential()
        
        current_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layer = Dense(current_dim, hidden_dim)
            # He initialization for ReLU
            layer.weights = np.random.randn(current_dim, hidden_dim) * np.sqrt(2.0 / current_dim)
            layer.biases = np.zeros((1, hidden_dim))
            
            self.network.add(layer)
            self.network.add(ReLU())
            current_dim = hidden_dim
        
        # Final layer to latent dimension - NO ACTIVATION for unrestricted latent space
        final_layer = Dense(current_dim, self.latent_dim)
        final_layer.weights = np.random.randn(current_dim, self.latent_dim) * np.sqrt(2.0 / current_dim)
        final_layer.biases = np.zeros((1, self.latent_dim))
        self.network.add(final_layer)
        
        print(f"Built encoder: {self.input_dim} → {' → '.join(map(str, self.hidden_dims))} → {self.latent_dim}")
    
    def forward(self, inputs):
        """
        Forward pass through encoder.
        
        Args:
            inputs (np.ndarray): Input data of shape (batch_size, input_dim)
            
        Returns:
            np.ndarray: Encoded features of shape (batch_size, latent_dim)
        """
        return self.network.forward(inputs)
    
    def predict(self, inputs):
        """
        Encode inputs to latent space (alias for forward).
        
        Args:
            inputs (np.ndarray): Input data to encode
            
        Returns:
            np.ndarray: Latent space representations
        """
        return self.forward(inputs)
    
    def get_network(self):
        """
        Get the underlying Sequential network.
        
        Returns:
            Sequential: The encoder network
        """
        return self.network


class DecoderWithLogits:
    """
    Decoder network for autoencoder architecture that outputs logits (no sigmoid).
    
    Reconstructs data from latent space back to original dimensions through
    a series of Dense layers with ReLU activations but NO final sigmoid.
    The output logits are meant to be used with BCEWithLogitsLoss for better
    numerical stability.
    
    Default architecture: 32 → 64 → 128 → 256 → 784 (logits)
    """
    
    def __init__(self, latent_dim=32, output_dim=784, hidden_dims=None):
        """
        Initialize decoder network that outputs logits.
        
        Args:
            latent_dim (int): Latent space dimension (default: 32)
            output_dim (int): Output dimension (default: 784 for MNIST)
            hidden_dims (list, optional): Hidden layer dimensions.
                                        If None, uses [64, 128, 256]
        """
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Default hidden dimensions if not specified (reverse of encoder)
        if hidden_dims is None:
            self.hidden_dims = [64, 128, 256]
        else:
            self.hidden_dims = hidden_dims
            
        self.network = None
        self._build_network()
    
    def _build_network(self):
        """Build decoder that outputs logits (no sigmoid activation)."""
        self.network = Sequential()
        
        current_dim = self.latent_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layer = Dense(current_dim, hidden_dim)
            # He initialization for ReLU
            layer.weights = np.random.randn(current_dim, hidden_dim) * np.sqrt(2.0 / current_dim)
            layer.biases = np.zeros((1, hidden_dim))
            
            self.network.add(layer)
            self.network.add(ReLU())
            current_dim = hidden_dim
        
        # Final layer outputs logits (NO sigmoid activation)
        final_layer = Dense(current_dim, self.output_dim)
        # Xavier/Glorot initialization for the final layer
        final_layer.weights = np.random.randn(current_dim, self.output_dim) * np.sqrt(2.0 / (current_dim + self.output_dim))
        final_layer.biases = np.zeros((1, self.output_dim))
        
        self.network.add(final_layer)
        # NO sigmoid activation here - outputs raw logits
        
        print(f"Built decoder with logits: {self.latent_dim} → {' → '.join(map(str, self.hidden_dims))} → {self.output_dim} (logits)")
    
    def forward(self, latent_features):
        """
        Forward pass through decoder.
        
        Args:
            latent_features (np.ndarray): Latent features of shape (batch_size, latent_dim)
            
        Returns:
            np.ndarray: Raw logits of shape (batch_size, output_dim)
        """
        return self.network.forward(latent_features)
    
    def predict(self, latent_features):
        """
        Decode latent features to logits (alias for forward).
        
        Args:
            latent_features (np.ndarray): Latent space features
            
        Returns:
            np.ndarray: Raw logits
        """
        return self.forward(latent_features)
    
    def predict_probabilities(self, latent_features):
        """
        Decode latent features and apply sigmoid to get probabilities.
        
        Args:
            latent_features (np.ndarray): Latent space features
            
        Returns:
            np.ndarray: Probabilities (after sigmoid)
        """
        logits = self.forward(latent_features)
        # Apply sigmoid to convert logits to probabilities
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))  # Clip to prevent overflow
        return probabilities
    
    def get_network(self):
        """
        Get the underlying Sequential network.
        
        Returns:
            Sequential: The decoder network
        """
        return self.network


class Decoder:
    """
    Decoder network for autoencoder architecture.
    
    Reconstructs data from latent space back to original dimensions through
    a series of Dense layers with ReLU activations and Sigmoid output.
    
    Default architecture: 32 → 64 → 128 → 256 → 784
    
    CRITICAL FIX: Uses larger weight initialization in final layer to prevent sigmoid saturation.
    """
    
    def __init__(self, latent_dim=32, output_dim=784, hidden_dims=None):
        """
        Initialize decoder network.
        
        Args:
            latent_dim (int): Latent space dimension (default: 32)
            output_dim (int): Output dimension (default: 784 for MNIST)
            hidden_dims (list, optional): Hidden layer dimensions.
                                        If None, uses [64, 128, 256]
        """
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Default hidden dimensions if not specified (reverse of encoder)
        if hidden_dims is None:
            self.hidden_dims = [64, 128, 256]
        else:
            self.hidden_dims = hidden_dims
            
        self.network = None
        self._build_network()
    
    def _build_network(self):
        """Build decoder with CRITICAL fix for sigmoid saturation."""
        self.network = Sequential()
        
        current_dim = self.latent_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layer = Dense(current_dim, hidden_dim)
            # He initialization for ReLU
            layer.weights = np.random.randn(current_dim, hidden_dim) * np.sqrt(2.0 / current_dim)
            layer.biases = np.zeros((1, hidden_dim))
            
            self.network.add(layer)
            self.network.add(ReLU())
            current_dim = hidden_dim
        
        # CRITICAL FIX: Final layer with larger weights to prevent sigmoid saturation
        final_layer = Dense(current_dim, self.output_dim)
        # Use larger initialization (0.5 instead of 0.05) to ensure sigmoid gets wide range of inputs
        final_layer.weights = np.random.randn(current_dim, self.output_dim) * 0.5  # LARGER WEIGHTS!
        final_layer.biases = np.zeros((1, self.output_dim))
        
        self.network.add(final_layer)
        self.network.add(Sigmoid())
        
        print(f"Built decoder: {self.latent_dim} → {' → '.join(map(str, self.hidden_dims))} → {self.output_dim} (FIXED sigmoid)")
    
    def forward(self, latent_features):
        """
        Forward pass through decoder.
        
        Args:
            latent_features (np.ndarray): Latent features of shape (batch_size, latent_dim)
            
        Returns:
            np.ndarray: Reconstructed data of shape (batch_size, output_dim)
        """
        return self.network.forward(latent_features)
    
    def predict(self, latent_features):
        """
        Decode latent features to original space (alias for forward).
        
        Args:
            latent_features (np.ndarray): Latent space features
            
        Returns:
            np.ndarray: Reconstructed data
        """
        return self.forward(latent_features)
    
    def get_network(self):
        """
        Get the underlying Sequential network.
        
        Returns:
            Sequential: The decoder network
        """
        return self.network


class Autoencoder:
    """
    Complete autoencoder combining encoder and decoder networks with sigmoid saturation fix.
    
    Provides training functionality with MSE loss for unsupervised learning
    where input equals target output for reconstruction tasks.
    """
    
    def __init__(self, encoder=None, decoder=None, latent_dim=32):
        """
        Initialize autoencoder with encoder and decoder networks.
        
        Args:
            encoder (Encoder, optional): Pre-built encoder network
            decoder (Decoder, optional): Pre-built decoder network
            latent_dim (int): Latent dimension if building networks from scratch
        """
        self.latent_dim = latent_dim
        
        # Use provided networks or create new ones
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = Encoder(latent_dim=latent_dim)
            
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = Decoder(latent_dim=latent_dim)
            
        # Combined network for end-to-end training
        self.network = None
        self._build_combined_network()
        
        # Training state
        self.is_trained = False
        self.training_history = None
    
    def _build_combined_network(self):
        """Build combined network with proper layer copying."""
        self.network = Sequential()
        
        # Copy encoder layers
        for layer in self.encoder.get_network().layers:
            if hasattr(layer, 'weights'):
                new_layer = Dense(layer.weights.shape[0], layer.weights.shape[1])
                new_layer.weights = layer.weights.copy()
                new_layer.biases = layer.biases.copy()
                self.network.add(new_layer)
            else:
                new_activation = type(layer)()
                self.network.add(new_activation)
        
        # Copy decoder layers
        for layer in self.decoder.get_network().layers:
            if hasattr(layer, 'weights'):
                new_layer = Dense(layer.weights.shape[0], layer.weights.shape[1])
                new_layer.weights = layer.weights.copy()
                new_layer.biases = layer.biases.copy()
                self.network.add(new_layer)
            else:
                new_activation = type(layer)()
                self.network.add(new_activation)
        
        print("Built complete autoencoder network")
    
    def forward(self, inputs):
        """
        Forward pass through complete autoencoder.
        
        Args:
            inputs (np.ndarray): Input data
            
        Returns:
            np.ndarray: Reconstructed data
        """
        return self.network.forward(inputs)
    
    def encode(self, inputs):
        """
        Encode inputs to latent space.
        
        Args:
            inputs (np.ndarray): Input data to encode
            
        Returns:
            np.ndarray: Latent space representations
        """
        return self.encoder.forward(inputs)
    
    def decode(self, latent_features):
        """
        Decode latent features back to original space.
        
        Args:
            latent_features (np.ndarray): Latent space features
            
        Returns:
            np.ndarray: Reconstructed data
        """
        return self.decoder.forward(latent_features)
    
    def reconstruct(self, inputs):
        """
        Reconstruct input data (encode then decode).
        
        Args:
            inputs (np.ndarray): Input data to reconstruct
            
        Returns:
            np.ndarray: Reconstructed data
        """
        return self.forward(inputs)
    
    def train(self, X_train, X_val=None, epochs=100, batch_size=32, learning_rate=0.01, 
              print_interval=10, validation_interval=10):
        """
        Train the autoencoder with optimal settings for sharp reconstructions.
        
        For autoencoders, the input data serves as both input and target
        (unsupervised reconstruction task).
        
        Args:
            X_train (np.ndarray): Training data
            X_val (np.ndarray, optional): Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for SGD optimizer (default: 0.01 for better convergence)
            print_interval (int): Print progress every N epochs
            validation_interval (int): Evaluate validation every N epochs
            
        Returns:
            dict: Training history with losses and metrics
        """
        # Create loss function and optimizer
        loss_fn = MSELoss()
        optimizer = SGD(learning_rate=learning_rate)
        
        # For autoencoder, input and target are the same (reconstruction task)
        y_train = X_train.copy()
        
        print(f"Training autoencoder for {epochs} epochs...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Latent dimension: {self.latent_dim}")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        
        # Check initial output range to monitor sigmoid saturation
        initial_output = self.network.forward(X_train[:5])
        print(f"Initial output range: [{initial_output.min():.3f}, {initial_output.max():.3f}]")
        
        # Initialize training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'epochs': [],
            'reconstruction_quality': []
        }
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        n_samples = X_train.shape[0]
        
        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:batch_end]
                y_batch = y_train_shuffled[i:batch_end]
                
                # Perform training step
                batch_loss = self.network.train_step(X_batch, y_batch, loss_fn, optimizer)
                epoch_losses.append(batch_loss)
            
            # Calculate average training loss for the epoch
            avg_train_loss = np.mean(epoch_losses)
            history['train_losses'].append(avg_train_loss)
            history['epochs'].append(epoch)
            
            # Validation evaluation
            val_loss = None
            if X_val is not None and (epoch + 1) % validation_interval == 0:
                val_predictions = self.network.predict(X_val)
                val_loss = loss_fn.forward(val_predictions, X_val)
                history['val_losses'].append(val_loss)
            
            # Calculate reconstruction quality metric (MSE)
            if (epoch + 1) % validation_interval == 0:
                sample_reconstructions = self.network.predict(X_train[:100])
                reconstruction_mse = np.mean((X_train[:100] - sample_reconstructions) ** 2)
                history['reconstruction_quality'].append(reconstruction_mse)
            
            # Print progress with range monitoring to detect saturation
            if (epoch + 1) % print_interval == 0 or epoch == 0:
                sample_recon = self.network.predict(X_train[:5])
                sample_latent = self.encode(X_train[:5])
                
                progress_msg = f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}"
                if val_loss is not None:
                    progress_msg += f", Val Loss: {val_loss:.6f}"
                progress_msg += f", Recon: [{sample_recon.min():.3f}, {sample_recon.max():.3f}]"
                progress_msg += f", Latent: [{sample_latent.min():.3f}, {sample_latent.max():.3f}]"
                print(progress_msg)
        
        self.is_trained = True
        self.training_history = history
        
        print("Autoencoder training completed!")
        
        # Final evaluation
        final_train_loss = history['train_losses'][-1]
        print(f"Final training loss: {final_train_loss:.6f}")
        
        # Final output range check
        final_output = self.network.forward(X_train[:5])
        print(f"Final output range: [{final_output.min():.3f}, {final_output.max():.3f}]")
        
        if X_val is not None:
            final_val_predictions = self.network.predict(X_val)
            final_val_loss = loss_fn.forward(final_val_predictions, X_val)
            print(f"Final validation loss: {final_val_loss:.6f}")
            history['final_val_loss'] = final_val_loss
        
        return history
    
    def evaluate_reconstruction_quality(self, X_test, n_samples=100):
        """
        Evaluate reconstruction quality on test data with comprehensive metrics.
        
        Args:
            X_test (np.ndarray): Test data
            n_samples (int): Number of samples to evaluate
            
        Returns:
            dict: Reconstruction quality metrics
        """
        if not self.is_trained:
            print("Warning: Autoencoder not trained yet")
        
        # Select random samples
        n_samples = min(n_samples, X_test.shape[0])
        indices = np.random.choice(X_test.shape[0], n_samples, replace=False)
        test_samples = X_test[indices]
        
        # Get reconstructions
        reconstructions = self.network.predict(test_samples)
        
        # Calculate metrics
        mse = np.mean((test_samples - reconstructions) ** 2)
        mae = np.mean(np.abs(test_samples - reconstructions))
        
        # Per-sample reconstruction error
        per_sample_mse = np.mean((test_samples - reconstructions) ** 2, axis=1)
        
        # Range analysis
        recon_min, recon_max = reconstructions.min(), reconstructions.max()
        input_min, input_max = test_samples.min(), test_samples.max()
        
        # Dynamic range preservation (key metric for detecting sigmoid saturation)
        input_range = input_max - input_min
        recon_range = recon_max - recon_min
        range_preservation = recon_range / input_range if input_range > 0 else 0
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'per_sample_mse': per_sample_mse,
            'mean_per_sample_mse': np.mean(per_sample_mse),
            'std_per_sample_mse': np.std(per_sample_mse),
            'n_samples_evaluated': n_samples,
            'reconstruction_range': (recon_min, recon_max),
            'input_range': (input_min, input_max),
            'range_preservation': range_preservation
        }
        
        print(f"Reconstruction Quality Metrics (n={n_samples}):")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Input range: [{input_min:.3f}, {input_max:.3f}]")
        print(f"  Reconstruction range: [{recon_min:.3f}, {recon_max:.3f}]")
        print(f"  Range preservation: {range_preservation:.4f}")
        
        # Quality assessment
        if range_preservation > 0.7:
            print("  ✅ Excellent dynamic range preservation")
        elif range_preservation > 0.5:
            print("  ⚠️  Good dynamic range preservation")
        else:
            print("  ❌ Poor dynamic range preservation - check for sigmoid saturation")
        
        return metrics
    
    def get_training_history(self):
        """
        Get training history.
        
        Returns:
            dict: Training history or None if not trained
        """
        return self.training_history
    
    def get_encoder(self):
        """
        Get the encoder network for feature extraction.
        
        Returns:
            Encoder: The encoder network
        """
        return self.encoder
    
    def get_decoder(self):
        """
        Get the decoder network.
        
        Returns:
            Decoder: The decoder network
        """
        return self.decoder


def create_encoder(input_dim=784, latent_dim=32, hidden_dims=None):
    """
    Convenience function to create an encoder network.
    
    Args:
        input_dim (int): Input dimension
        latent_dim (int): Latent space dimension
        hidden_dims (list, optional): Hidden layer dimensions
        
    Returns:
        Encoder: Configured encoder network
    """
    return Encoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)


def create_decoder(latent_dim=32, output_dim=784, hidden_dims=None):
    """
    Convenience function to create a decoder network.
    
    Args:
        latent_dim (int): Latent space dimension
        output_dim (int): Output dimension
        hidden_dims (list, optional): Hidden layer dimensions
        
    Returns:
        Decoder: Configured decoder network
    """
    return Decoder(latent_dim=latent_dim, output_dim=output_dim, hidden_dims=hidden_dims)


def create_autoencoder(latent_dim=32, input_dim=784, encoder_hidden_dims=None, 
                      decoder_hidden_dims=None):
    """
    Convenience function to create a complete autoencoder with sigmoid saturation fix.
    
    Args:
        latent_dim (int): Latent space dimension (default: 32)
        input_dim (int): Input dimension (default: 784 for MNIST)
        encoder_hidden_dims (list, optional): Encoder hidden layer dimensions
        decoder_hidden_dims (list, optional): Decoder hidden layer dimensions
        
    Returns:
        Autoencoder: Configured autoencoder ready for training with sharp reconstructions
    """
    # Create encoder and decoder
    encoder = create_encoder(
        input_dim=input_dim, 
        latent_dim=latent_dim, 
        hidden_dims=encoder_hidden_dims
    )
    
    decoder = create_decoder(
        latent_dim=latent_dim, 
        output_dim=input_dim, 
        hidden_dims=decoder_hidden_dims
    )
    
    # Create autoencoder
    autoencoder = Autoencoder(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    
    return autoencoder


class AutoencoderWithLogits:
    """
    Autoencoder that outputs logits and uses BCE with logits loss for better numerical stability.
    
    This version removes the sigmoid activation from the decoder output and uses
    BCEWithLogitsLoss instead of MSELoss, which provides better numerical stability
    and can lead to improved training dynamics.
    """
    
    def __init__(self, encoder=None, decoder=None, latent_dim=32):
        """
        Initialize autoencoder with logits output.
        
        Args:
            encoder (Encoder, optional): Pre-built encoder network
            decoder (DecoderWithLogits, optional): Pre-built decoder network that outputs logits
            latent_dim (int): Latent dimension if building networks from scratch
        """
        self.latent_dim = latent_dim
        
        # Use provided networks or create new ones
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = Encoder(latent_dim=latent_dim)
            
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = DecoderWithLogits(latent_dim=latent_dim)
            
        # Combined network for end-to-end training
        self.network = None
        self._build_combined_network()
        
        # Training state
        self.is_trained = False
        self.training_history = None
    
    def _build_combined_network(self):
        """Build combined network with proper layer copying."""
        self.network = Sequential()
        
        # Copy encoder layers
        for layer in self.encoder.get_network().layers:
            if hasattr(layer, 'weights'):
                new_layer = Dense(layer.weights.shape[0], layer.weights.shape[1])
                new_layer.weights = layer.weights.copy()
                new_layer.biases = layer.biases.copy()
                self.network.add(new_layer)
            else:
                new_activation = type(layer)()
                self.network.add(new_activation)
        
        # Copy decoder layers
        for layer in self.decoder.get_network().layers:
            if hasattr(layer, 'weights'):
                new_layer = Dense(layer.weights.shape[0], layer.weights.shape[1])
                new_layer.weights = layer.weights.copy()
                new_layer.biases = layer.biases.copy()
                self.network.add(new_layer)
            else:
                new_activation = type(layer)()
                self.network.add(new_activation)
        
        print("Built complete autoencoder network with logits output")
    
    def forward(self, inputs):
        """
        Forward pass through complete autoencoder (returns logits).
        
        Args:
            inputs (np.ndarray): Input data
            
        Returns:
            np.ndarray: Reconstructed logits (before sigmoid)
        """
        return self.network.forward(inputs)
    
    def encode(self, inputs):
        """
        Encode inputs to latent space.
        
        Args:
            inputs (np.ndarray): Input data to encode
            
        Returns:
            np.ndarray: Latent space representations
        """
        return self.encoder.forward(inputs)
    
    def decode(self, latent_features):
        """
        Decode latent features back to logits.
        
        Args:
            latent_features (np.ndarray): Latent space features
            
        Returns:
            np.ndarray: Reconstructed logits (before sigmoid)
        """
        return self.decoder.forward(latent_features)
    
    def reconstruct_logits(self, inputs):
        """
        Reconstruct input data as logits (encode then decode).
        
        Args:
            inputs (np.ndarray): Input data to reconstruct
            
        Returns:
            np.ndarray: Reconstructed logits
        """
        return self.forward(inputs)
    
    def reconstruct_probabilities(self, inputs):
        """
        Reconstruct input data as probabilities (apply sigmoid to logits).
        
        Args:
            inputs (np.ndarray): Input data to reconstruct
            
        Returns:
            np.ndarray: Reconstructed probabilities (after sigmoid)
        """
        logits = self.forward(inputs)
        # Apply sigmoid to convert logits to probabilities
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))  # Clip to prevent overflow
        return probabilities
    
    def train(self, X_train, X_val=None, epochs=100, batch_size=32, learning_rate=0.01, 
              print_interval=10, validation_interval=10):
        """
        Train the autoencoder with BCE with logits loss for better numerical stability.
        
        Args:
            X_train (np.ndarray): Training data (should be in [0, 1] range)
            X_val (np.ndarray, optional): Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for SGD optimizer
            print_interval (int): Print progress every N epochs
            validation_interval (int): Evaluate validation every N epochs
            
        Returns:
            dict: Training history with losses and metrics
        """
        # Create loss function and optimizer
        loss_fn = BCEWithLogitsLoss()
        optimizer = SGD(learning_rate=learning_rate)
        
        # For autoencoder, input and target are the same (reconstruction task)
        y_train = X_train.copy()
        
        print(f"Training autoencoder with logits for {epochs} epochs...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Latent dimension: {self.latent_dim}")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        print(f"Loss function: BCE with Logits (numerically stable)")
        
        # Check initial output range
        initial_logits = self.network.forward(X_train[:5])
        initial_probs = 1 / (1 + np.exp(-np.clip(initial_logits, -500, 500)))
        print(f"Initial logits range: [{initial_logits.min():.3f}, {initial_logits.max():.3f}]")
        print(f"Initial probabilities range: [{initial_probs.min():.3f}, {initial_probs.max():.3f}]")
        
        # Initialize training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'epochs': [],
            'reconstruction_quality': []
        }
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        n_samples = X_train.shape[0]
        
        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:batch_end]
                y_batch = y_train_shuffled[i:batch_end]
                
                # Perform training step
                batch_loss = self.network.train_step(X_batch, y_batch, loss_fn, optimizer)
                epoch_losses.append(batch_loss)
            
            # Calculate average training loss for the epoch
            avg_train_loss = np.mean(epoch_losses)
            history['train_losses'].append(avg_train_loss)
            history['epochs'].append(epoch)
            
            # Validation evaluation
            val_loss = None
            if X_val is not None and (epoch + 1) % validation_interval == 0:
                val_logits = self.network.predict(X_val)
                val_loss = loss_fn.forward(val_logits, X_val)
                history['val_losses'].append(val_loss)
            
            # Calculate reconstruction quality metric (using probabilities)
            if (epoch + 1) % validation_interval == 0:
                sample_logits = self.network.predict(X_train[:100])
                sample_probs = 1 / (1 + np.exp(-np.clip(sample_logits, -500, 500)))
                reconstruction_mse = np.mean((X_train[:100] - sample_probs) ** 2)
                history['reconstruction_quality'].append(reconstruction_mse)
            
            # Print progress with logits and probabilities monitoring
            if (epoch + 1) % print_interval == 0 or epoch == 0:
                sample_logits = self.network.predict(X_train[:5])
                sample_probs = 1 / (1 + np.exp(-np.clip(sample_logits, -500, 500)))
                sample_latent = self.encode(X_train[:5])
                
                progress_msg = f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}"
                if val_loss is not None:
                    progress_msg += f", Val Loss: {val_loss:.6f}"
                progress_msg += f", Logits: [{sample_logits.min():.3f}, {sample_logits.max():.3f}]"
                progress_msg += f", Probs: [{sample_probs.min():.3f}, {sample_probs.max():.3f}]"
                progress_msg += f", Latent: [{sample_latent.min():.3f}, {sample_latent.max():.3f}]"
                print(progress_msg)
        
        self.is_trained = True
        self.training_history = history
        
        print("Autoencoder with logits training completed!")
        
        # Final evaluation
        final_train_loss = history['train_losses'][-1]
        print(f"Final training loss: {final_train_loss:.6f}")
        
        # Final output range check
        final_logits = self.network.forward(X_train[:5])
        final_probs = 1 / (1 + np.exp(-np.clip(final_logits, -500, 500)))
        print(f"Final logits range: [{final_logits.min():.3f}, {final_logits.max():.3f}]")
        print(f"Final probabilities range: [{final_probs.min():.3f}, {final_probs.max():.3f}]")
        
        if X_val is not None:
            final_val_logits = self.network.predict(X_val)
            final_val_loss = loss_fn.forward(final_val_logits, X_val)
            print(f"Final validation loss: {final_val_loss:.6f}")
            history['final_val_loss'] = final_val_loss
        
        return history
    
    def evaluate_reconstruction_quality(self, X_test, n_samples=100):
        """
        Evaluate reconstruction quality on test data using probabilities.
        
        Args:
            X_test (np.ndarray): Test data
            n_samples (int): Number of samples to evaluate
            
        Returns:
            dict: Reconstruction quality metrics
        """
        if not self.is_trained:
            print("Warning: Autoencoder not trained yet")
        
        # Select random samples
        n_samples = min(n_samples, X_test.shape[0])
        indices = np.random.choice(X_test.shape[0], n_samples, replace=False)
        test_samples = X_test[indices]
        
        # Get reconstructions as probabilities
        logits = self.network.predict(test_samples)
        reconstructions = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        
        # Calculate metrics
        mse = np.mean((test_samples - reconstructions) ** 2)
        mae = np.mean(np.abs(test_samples - reconstructions))
        
        # BCE loss
        bce_loss = BCEWithLogitsLoss().forward(logits, test_samples)
        
        # Per-sample reconstruction error
        per_sample_mse = np.mean((test_samples - reconstructions) ** 2, axis=1)
        
        # Range analysis
        recon_min, recon_max = reconstructions.min(), reconstructions.max()
        input_min, input_max = test_samples.min(), test_samples.max()
        logits_min, logits_max = logits.min(), logits.max()
        
        # Dynamic range preservation
        input_range = input_max - input_min
        recon_range = recon_max - recon_min
        range_preservation = recon_range / input_range if input_range > 0 else 0
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'bce_loss': bce_loss,
            'per_sample_mse': per_sample_mse,
            'mean_per_sample_mse': np.mean(per_sample_mse),
            'std_per_sample_mse': np.std(per_sample_mse),
            'n_samples_evaluated': n_samples,
            'reconstruction_range': (recon_min, recon_max),
            'input_range': (input_min, input_max),
            'logits_range': (logits_min, logits_max),
            'range_preservation': range_preservation
        }
        
        print(f"Reconstruction Quality Metrics (n={n_samples}):")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  BCE Loss: {bce_loss:.6f}")
        print(f"  Input range: [{input_min:.3f}, {input_max:.3f}]")
        print(f"  Reconstruction range: [{recon_min:.3f}, {recon_max:.3f}]")
        print(f"  Logits range: [{logits_min:.3f}, {logits_max:.3f}]")
        print(f"  Range preservation: {range_preservation:.4f}")
        
        # Quality assessment
        if range_preservation > 0.8:
            print("  ✅ Excellent dynamic range preservation")
        elif range_preservation > 0.6:
            print("  ⚠️  Good dynamic range preservation")
        else:
            print("  ❌ Poor dynamic range preservation")
        
        return metrics
    
    def get_training_history(self):
        """Get training history."""
        return self.training_history
    
    def get_encoder(self):
        """Get the encoder network."""
        return self.encoder
    
    def get_decoder(self):
        """Get the decoder network."""
        return self.decoder


def create_autoencoder_with_logits(latent_dim=32, input_dim=784, encoder_hidden_dims=None, 
                                  decoder_hidden_dims=None):
    """
    Convenience function to create an autoencoder with logits output and BCE with logits loss.
    
    This version provides better numerical stability by avoiding the sigmoid activation
    in the decoder and using BCE with logits loss instead.
    
    Args:
        latent_dim (int): Latent space dimension (default: 32)
        input_dim (int): Input dimension (default: 784 for MNIST)
        encoder_hidden_dims (list, optional): Encoder hidden layer dimensions
        decoder_hidden_dims (list, optional): Decoder hidden layer dimensions
        
    Returns:
        AutoencoderWithLogits: Configured autoencoder with logits output
    """
    # Create encoder and decoder with logits
    encoder = create_encoder(
        input_dim=input_dim, 
        latent_dim=latent_dim, 
        hidden_dims=encoder_hidden_dims
    )
    
    decoder = DecoderWithLogits(
        latent_dim=latent_dim, 
        output_dim=input_dim, 
        hidden_dims=decoder_hidden_dims
    )
    
    # Create autoencoder with logits
    autoencoder = AutoencoderWithLogits(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    
    return autoencoder


def train_autoencoder_on_mnist(latent_dim=32, epochs=100, learning_rate=0.01, 
                              batch_size=32, validation_split=0.1):
    """
    Convenience function to train an autoencoder on MNIST data with optimal settings.
    
    Args:
        latent_dim (int): Latent space dimension
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for training (default: 0.01 for better convergence)
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data to use for validation
        
    Returns:
        tuple: (trained_autoencoder, training_history, data_info)
    """
    try:
        from .data_utils import MNISTDataLoader
        
        # Load and preprocess MNIST data
        print("Loading MNIST data...")
        data_loader = MNISTDataLoader()
        X_train, X_test, y_train, y_test = data_loader.load_data()
        X_train_processed, X_test_processed = data_loader.preprocess_data()
        
        # Split training data for validation
        n_val = int(len(X_train_processed) * validation_split)
        X_val = X_train_processed[:n_val]
        X_train_final = X_train_processed[n_val:]
        
        print(f"Training set size: {X_train_final.shape[0]}")
        print(f"Validation set size: {X_val.shape[0]}")
        print(f"Test set size: {X_test_processed.shape[0]}")
        
        # Create and train autoencoder
        autoencoder = create_autoencoder(latent_dim=latent_dim)
        history = autoencoder.train(
            X_train_final, X_val, 
            epochs=epochs, 
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
        # Evaluate on test set
        test_metrics = autoencoder.evaluate_reconstruction_quality(X_test_processed)
        
        data_info = {
            'train_shape': X_train_final.shape,
            'val_shape': X_val.shape,
            'test_shape': X_test_processed.shape,
            'test_metrics': test_metrics
        }
        
        return autoencoder, history, data_info
        
    except ImportError:
        print("Error: MNISTDataLoader not available. Please ensure data_utils module is properly implemented.")
        return None, None, None


def train_autoencoder_with_logits_on_mnist(latent_dim=32, epochs=100, learning_rate=0.01, 
                                          batch_size=32, validation_split=0.1):
    """
    Convenience function to train an autoencoder with logits on MNIST data.
    
    This version uses BCE with logits loss for better numerical stability.
    
    Args:
        latent_dim (int): Latent space dimension
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for training
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data to use for validation
        
    Returns:
        tuple: (trained_autoencoder, training_history, data_info)
    """
    try:
        from .data_utils import MNISTDataLoader
        
        # Load and preprocess MNIST data
        print("Loading MNIST data...")
        data_loader = MNISTDataLoader()
        X_train, X_test, y_train, y_test = data_loader.load_data()
        X_train_processed, X_test_processed = data_loader.preprocess_data()
        
        # Split training data for validation
        n_val = int(len(X_train_processed) * validation_split)
        X_val = X_train_processed[:n_val]
        X_train_final = X_train_processed[n_val:]
        
        print(f"Training set size: {X_train_final.shape[0]}")
        print(f"Validation set size: {X_val.shape[0]}")
        print(f"Test set size: {X_test_processed.shape[0]}")
        
        # Create and train autoencoder with logits
        autoencoder = create_autoencoder_with_logits(latent_dim=latent_dim)
        history = autoencoder.train(
            X_train_final, X_val, 
            epochs=epochs, 
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
        # Evaluate on test set
        test_metrics = autoencoder.evaluate_reconstruction_quality(X_test_processed)
        
        data_info = {
            'train_shape': X_train_final.shape,
            'val_shape': X_val.shape,
            'test_shape': X_test_processed.shape,
            'test_metrics': test_metrics
        }
        
        return autoencoder, history, data_info
        
    except ImportError:
        print("Error: MNISTDataLoader not available. Please ensure data_utils module is properly implemented.")
        return None, None, None