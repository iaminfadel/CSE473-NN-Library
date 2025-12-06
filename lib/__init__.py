"""
Neural Network Library

A from-scratch implementation of neural networks using only NumPy.
This library provides modular components for building and training neural networks.
"""

__version__ = "1.0.0"
__author__ = "Neural Network Library Team"

# Import main components for easy access
from .layers import Layer, Dense
from .activations import ReLU, Sigmoid, Tanh, Softmax
from .losses import MSELoss
from .optimizer import SGD
from .network import Sequential
from .gradient_checker import GradientChecker, check_gradients
from .autoencoder import (
    Encoder, Decoder, Autoencoder,
    create_encoder, create_decoder, create_autoencoder,
    train_autoencoder_on_mnist
)
from .data_utils import (
    MNISTDataLoader, BatchIterator, VisualizationUtils,
    create_autoencoder_data, prepare_svm_data,
    show_mnist_samples, show_reconstruction_comparison, plot_loss_curve
)

__all__ = [
    'Layer', 'Dense',
    'ReLU', 'Sigmoid', 'Tanh', 'Softmax',
    'MSELoss',
    'SGD',
    'Sequential',
    'GradientChecker', 'check_gradients',
    'Encoder', 'Decoder', 'Autoencoder',
    'create_encoder', 'create_decoder', 'create_autoencoder',
    'train_autoencoder_on_mnist',
    'MNISTDataLoader', 'BatchIterator', 'VisualizationUtils',
    'create_autoencoder_data', 'prepare_svm_data',
    'show_mnist_samples', 'show_reconstruction_comparison', 'plot_loss_curve'
]