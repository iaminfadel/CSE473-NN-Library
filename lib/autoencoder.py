"""
Autoencoder architecture implementations.

This module provides modular encoder and decoder networks for building
autoencoders with configurable latent dimensions.
"""

import numpy as np
from .network import Sequential
from .layers import Dense
from .activations import ReLU, Sigmoid
from .losses import MSELoss
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
        """Build the encoder network architecture."""
        self.network = Sequential()
        
        # Build layers progressively
        current_dim = self.input_dim
        
        # Add hidden layers with ReLU activations
        for hidden_dim in self.hidden_dims:
            self.network.add(Dense(current_dim, hidden_dim))
            self.network.add(ReLU())
            current_dim = hidden_dim
        
        # Add final layer to latent dimension
        self.network.add(Dense(current_dim, self.latent_dim))
        self.network.add(ReLU())
        
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


class Decoder:
    """
    Decoder network for autoencoder architecture.
    
    Reconstructs data from latent space back to original dimensions through
    a series of Dense layers with ReLU activations and Sigmoid output.
    
    Default architecture: 32 → 64 → 128 → 256 → 784
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
        """Build the decoder network architecture."""
        self.network = Sequential()
        
        # Build layers progressively
        current_dim = self.latent_dim
        
        # Add hidden layers with ReLU activations
        for hidden_dim in self.hidden_dims:
            self.network.add(Dense(current_dim, hidden_dim))
            self.network.add(ReLU())
            current_dim = hidden_dim
        
        # Add final layer to output dimension with Sigmoid activation
        # Sigmoid ensures output is in [0, 1] range for normalized MNIST data
        self.network.add(Dense(current_dim, self.output_dim))
        self.network.add(Sigmoid())
        
        print(f"Built decoder: {self.latent_dim} → {' → '.join(map(str, self.hidden_dims))} → {self.output_dim}")
    
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
    Complete autoencoder combining encoder and decoder networks.
    
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
        """Build the complete autoencoder by combining encoder and decoder."""
        self.network = Sequential()
        
        # Add all encoder layers
        for layer in self.encoder.get_network().layers:
            self.network.add(layer)
            
        # Add all decoder layers
        for layer in self.decoder.get_network().layers:
            self.network.add(layer)
            
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
    
    def train(self, X_train, X_val=None, epochs=50, batch_size=32, learning_rate=0.001, 
              print_interval=10, validation_interval=10):
        """
        Train the autoencoder with unsupervised learning.
        
        For autoencoders, the input data serves as both input and target
        (unsupervised reconstruction task).
        
        Args:
            X_train (np.ndarray): Training data
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
        loss_fn = MSELoss()
        optimizer = SGD(learning_rate=learning_rate)
        
        # For autoencoder, input and target are the same (reconstruction task)
        y_train = X_train.copy()
        
        print(f"Training autoencoder for {epochs} epochs...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Latent dimension: {self.latent_dim}")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        
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
            
            # Print progress
            if (epoch + 1) % print_interval == 0 or epoch == 0:
                progress_msg = f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}"
                if val_loss is not None:
                    progress_msg += f", Val Loss: {val_loss:.6f}"
                print(progress_msg)
        
        self.is_trained = True
        self.training_history = history
        
        print("Autoencoder training completed!")
        
        # Final evaluation
        final_train_loss = history['train_losses'][-1]
        print(f"Final training loss: {final_train_loss:.6f}")
        
        if X_val is not None:
            final_val_predictions = self.network.predict(X_val)
            final_val_loss = loss_fn.forward(final_val_predictions, X_val)
            print(f"Final validation loss: {final_val_loss:.6f}")
            history['final_val_loss'] = final_val_loss
        
        return history
    
    def evaluate_reconstruction_quality(self, X_test, n_samples=100):
        """
        Evaluate reconstruction quality on test data.
        
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
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'per_sample_mse': per_sample_mse,
            'mean_per_sample_mse': np.mean(per_sample_mse),
            'std_per_sample_mse': np.std(per_sample_mse),
            'n_samples_evaluated': n_samples
        }
        
        print(f"Reconstruction Quality Metrics (n={n_samples}):")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Mean per-sample MSE: {metrics['mean_per_sample_mse']:.6f}")
        print(f"  Std per-sample MSE: {metrics['std_per_sample_mse']:.6f}")
        
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
    Convenience function to create a complete autoencoder.
    
    Args:
        latent_dim (int): Latent space dimension
        input_dim (int): Input dimension (default: 784 for MNIST)
        encoder_hidden_dims (list, optional): Encoder hidden layer dimensions
        decoder_hidden_dims (list, optional): Decoder hidden layer dimensions
        
    Returns:
        Autoencoder: Configured autoencoder ready for training
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


def train_autoencoder_on_mnist(latent_dim=32, epochs=50, learning_rate=0.001, 
                              batch_size=32, validation_split=0.1):
    """
    Convenience function to train an autoencoder on MNIST data.
    
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