"""
MNIST data loading and preprocessing utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class MNISTDataLoader:
    """
    MNIST dataset loader and preprocessor.
    
    Handles loading MNIST data from sklearn, normalization, flattening,
    and batch processing utilities.
    """
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_loaded = False
        
    def load_data(self, test_size=0.2, random_state=42):
        """
        Load MNIST dataset using sklearn.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Loading MNIST dataset...")
        
        # Load MNIST data from sklearn
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Loaded MNIST: {self.X_train.shape[0]} training samples, {self.X_test.shape[0]} test samples")
        self.is_loaded = True
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preprocess_data(self, normalize=True, flatten=True):
        """
        Preprocess MNIST data with normalization and flattening.
        
        Args:
            normalize (bool): Whether to normalize pixel values to [0, 1]
            flatten (bool): Whether to flatten images to 784-dimensional vectors
            
        Returns:
            tuple: (X_train_processed, X_test_processed)
        """
        if not self.is_loaded:
            raise ValueError("Data must be loaded first. Call load_data().")
            
        X_train_processed = self.X_train.copy()
        X_test_processed = self.X_test.copy()
        
        # Normalize pixel values to [0, 1] range
        if normalize:
            X_train_processed = X_train_processed.astype(np.float32) / 255.0
            X_test_processed = X_test_processed.astype(np.float32) / 255.0
            print("Normalized pixel values to [0, 1] range")
        
        # Flatten images from 28x28 to 784 dimensions
        if flatten:
            if len(X_train_processed.shape) > 2:
                X_train_processed = X_train_processed.reshape(X_train_processed.shape[0], -1)
                X_test_processed = X_test_processed.reshape(X_test_processed.shape[0], -1)
            print(f"Flattened images to {X_train_processed.shape[1]} dimensions")
        
        return X_train_processed, X_test_processed
    
    def get_sample_images(self, num_samples=10, from_test=True):
        """
        Get sample images for visualization.
        
        Args:
            num_samples (int): Number of sample images to return
            from_test (bool): Whether to sample from test set or training set
            
        Returns:
            tuple: (images, labels) - images in original 28x28 format
        """
        if not self.is_loaded:
            raise ValueError("Data must be loaded first. Call load_data().")
            
        if from_test:
            X, y = self.X_test, self.y_test
        else:
            X, y = self.X_train, self.y_train
            
        # Select random samples
        indices = np.random.choice(len(X), num_samples, replace=False)
        sample_images = X[indices]
        sample_labels = y[indices]
        
        # Reshape to 28x28 if flattened
        if len(sample_images.shape) == 2 and sample_images.shape[1] == 784:
            sample_images = sample_images.reshape(-1, 28, 28)
        
        # Normalize if needed
        if sample_images.max() > 1.0:
            sample_images = sample_images.astype(np.float32) / 255.0
            
        return sample_images, sample_labels


class BatchIterator:
    """
    Utility class for creating mini-batches from datasets.
    """
    
    def __init__(self, X, y=None, batch_size=32, shuffle=True):
        """
        Initialize batch iterator.
        
        Args:
            X (np.ndarray): Input data
            y (np.ndarray, optional): Target labels (None for unsupervised)
            batch_size (int): Size of each batch
            shuffle (bool): Whether to shuffle data before batching
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        
    def __iter__(self):
        """
        Iterator protocol implementation.
        
        Yields:
            tuple: (X_batch, y_batch) or just X_batch if y is None
        """
        # Create indices
        indices = np.arange(self.n_samples)
        
        # Shuffle if requested
        if self.shuffle:
            np.random.shuffle(indices)
            
        # Generate batches
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            X_batch = self.X[batch_indices]
            
            if self.y is not None:
                y_batch = self.y[batch_indices]
                yield X_batch, y_batch
            else:
                yield X_batch
                
    def __len__(self):
        """Return number of batches."""
        return self.n_batches


def create_autoencoder_data(X_train, X_test):
    """
    Prepare data for autoencoder training (unsupervised).
    
    For autoencoders, the input and target are the same (reconstruction task).
    
    Args:
        X_train (np.ndarray): Training input data
        X_test (np.ndarray): Test input data
        
    Returns:
        tuple: (X_train, X_train, X_test, X_test) - inputs and targets are identical
    """
    return X_train, X_train, X_test, X_test


def prepare_svm_data(encoder_model, X_train, y_train, X_test, y_test):
    """
    Extract latent features using trained encoder for SVM classification.
    
    Args:
        encoder_model: Trained encoder network
        X_train (np.ndarray): Training input data
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test input data
        y_test (np.ndarray): Test labels
        
    Returns:
        tuple: (X_train_latent, y_train, X_test_latent, y_test)
    """
    # Extract latent features using encoder
    X_train_latent = encoder_model.predict(X_train)
    X_test_latent = encoder_model.predict(X_test)
    
    print(f"Extracted latent features: {X_train_latent.shape[1]} dimensions")
    
    return X_train_latent, y_train, X_test_latent, y_test


class VisualizationUtils:
    """
    Utilities for visualizing MNIST images and training progress.
    """
    
    @staticmethod
    def display_images(images, labels=None, predictions=None, title="Images", 
                      figsize=(12, 8), max_images=10):
        """
        Display a grid of MNIST images.
        
        Args:
            images (np.ndarray): Images to display (N, 28, 28) or (N, 784)
            labels (np.ndarray, optional): True labels for images
            predictions (np.ndarray, optional): Predicted labels for images
            title (str): Title for the plot
            figsize (tuple): Figure size
            max_images (int): Maximum number of images to display
        """
        # Limit number of images
        num_images = min(len(images), max_images)
        images = images[:num_images]
        
        # Reshape if flattened
        if len(images.shape) == 2 and images.shape[1] == 784:
            images = images.reshape(-1, 28, 28)
        
        # Calculate grid dimensions
        cols = min(5, num_images)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Handle single subplot case
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
            
        for i in range(num_images):
            row = i // cols
            col = i % cols
            
            ax = axes[row, col]
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
            
            # Add labels if provided
            if labels is not None:
                if predictions is not None:
                    ax.set_title(f'True: {labels[i]}, Pred: {predictions[i]}')
                else:
                    ax.set_title(f'Label: {labels[i]}')
        
        # Hide unused subplots
        for i in range(num_images, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def display_original_vs_reconstructed(original_images, reconstructed_images, 
                                        num_pairs=5, figsize=(12, 6)):
        """
        Display original images alongside their reconstructions.
        
        Args:
            original_images (np.ndarray): Original images (N, 28, 28) or (N, 784)
            reconstructed_images (np.ndarray): Reconstructed images (N, 28, 28) or (N, 784)
            num_pairs (int): Number of image pairs to display
            figsize (tuple): Figure size
        """
        # Limit number of pairs
        num_pairs = min(num_pairs, len(original_images))
        
        # Reshape if flattened
        if len(original_images.shape) == 2 and original_images.shape[1] == 784:
            original_images = original_images.reshape(-1, 28, 28)
        if len(reconstructed_images.shape) == 2 and reconstructed_images.shape[1] == 784:
            reconstructed_images = reconstructed_images.reshape(-1, 28, 28)
        
        fig, axes = plt.subplots(2, num_pairs, figsize=figsize)
        fig.suptitle('Original vs Reconstructed Images', fontsize=16)
        
        for i in range(num_pairs):
            # Original image
            axes[0, i].imshow(original_images[i], cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed image
            axes[1, i].imshow(reconstructed_images[i], cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_loss(losses, title="Training Loss", figsize=(10, 6), 
                          xlabel="Epoch", ylabel="Loss"):
        """
        Plot training loss curve.
        
        Args:
            losses (list): List of loss values per epoch
            title (str): Plot title
            figsize (tuple): Figure size
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
        """
        plt.figure(figsize=figsize)
        plt.plot(losses, 'b-', linewidth=2)
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_multiple_metrics(metrics_dict, title="Training Metrics", figsize=(12, 6)):
        """
        Plot multiple training metrics on the same or separate subplots.
        
        Args:
            metrics_dict (dict): Dictionary with metric names as keys and lists of values
            title (str): Overall plot title
            figsize (tuple): Figure size
        """
        num_metrics = len(metrics_dict)
        
        if num_metrics == 1:
            # Single metric
            metric_name, values = list(metrics_dict.items())[0]
            VisualizationUtils.plot_training_loss(values, title=f"{title} - {metric_name}")
        else:
            # Multiple metrics
            fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
            fig.suptitle(title, fontsize=16)
            
            for i, (metric_name, values) in enumerate(metrics_dict.items()):
                ax = axes[i] if num_metrics > 1 else axes
                ax.plot(values, 'b-', linewidth=2)
                ax.set_title(metric_name)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def plot_reconstruction_quality(original_images, reconstructed_images, 
                                  mse_per_image=None, num_examples=10, figsize=(15, 8)):
        """
        Comprehensive visualization of reconstruction quality.
        
        Args:
            original_images (np.ndarray): Original images
            reconstructed_images (np.ndarray): Reconstructed images
            mse_per_image (np.ndarray, optional): MSE loss per image
            num_examples (int): Number of examples to show
            figsize (tuple): Figure size
        """
        # Limit examples
        num_examples = min(num_examples, len(original_images))
        
        # Reshape if needed
        if len(original_images.shape) == 2 and original_images.shape[1] == 784:
            original_images = original_images.reshape(-1, 28, 28)
        if len(reconstructed_images.shape) == 2 and reconstructed_images.shape[1] == 784:
            reconstructed_images = reconstructed_images.reshape(-1, 28, 28)
        
        # Create figure with subplots
        rows = 3 if mse_per_image is not None else 2
        fig, axes = plt.subplots(rows, num_examples, figsize=figsize)
        fig.suptitle('Reconstruction Quality Analysis', fontsize=16)
        
        for i in range(num_examples):
            # Original image
            axes[0, i].imshow(original_images[i], cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed image
            axes[1, i].imshow(reconstructed_images[i], cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
            
            # Difference/error visualization
            if mse_per_image is not None:
                diff = np.abs(original_images[i] - reconstructed_images[i])
                im = axes[2, i].imshow(diff, cmap='hot')
                axes[2, i].set_title(f'Error (MSE: {mse_per_image[i]:.4f})')
                axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_latent_space_2d(latent_features, labels, title="Latent Space Visualization", 
                           figsize=(10, 8), max_samples=1000):
        """
        Visualize 2D latent space (for 2D latent dimensions or PCA-reduced).
        
        Args:
            latent_features (np.ndarray): Latent space features (N, 2)
            labels (np.ndarray): Corresponding labels
            title (str): Plot title
            figsize (tuple): Figure size
            max_samples (int): Maximum samples to plot for performance
        """
        # Limit samples for performance
        if len(latent_features) > max_samples:
            indices = np.random.choice(len(latent_features), max_samples, replace=False)
            latent_features = latent_features[indices]
            labels = labels[indices]
        
        plt.figure(figsize=figsize)
        
        # Create scatter plot with different colors for each digit
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(latent_features[mask, 0], latent_features[mask, 1], 
                       c=[colors[i]], label=f'Digit {label}', alpha=0.6)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Convenience functions for common visualization tasks
def show_mnist_samples(data_loader, num_samples=10):
    """
    Quick function to display MNIST samples.
    
    Args:
        data_loader (MNISTDataLoader): Loaded MNIST data
        num_samples (int): Number of samples to show
    """
    images, labels = data_loader.get_sample_images(num_samples)
    VisualizationUtils.display_images(images, labels, title="MNIST Sample Images")


def show_reconstruction_comparison(original, reconstructed, num_pairs=5):
    """
    Quick function to compare original and reconstructed images.
    
    Args:
        original (np.ndarray): Original images
        reconstructed (np.ndarray): Reconstructed images
        num_pairs (int): Number of pairs to show
    """
    VisualizationUtils.display_original_vs_reconstructed(original, reconstructed, num_pairs)


def plot_loss_curve(losses, title="Training Loss"):
    """
    Quick function to plot training loss.
    
    Args:
        losses (list): Loss values
        title (str): Plot title
    """
    VisualizationUtils.plot_training_loss(losses, title=title)