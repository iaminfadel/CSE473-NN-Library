# Neural Network Library

A from-scratch implementation of neural networks using only Python and NumPy. This project demonstrates fundamental neural network concepts through three progressive applications: XOR problem solving, MNIST autoencoder implementation, and latent space feature extraction for SVM classification.

## Project Structure

```
.
├── .gitignore
├── README.md                  # Main project documentation
├── requirements.txt           # Dependencies (numpy, matplotlib, etc.)
├── lib/                       # Neural network library code
│   ├── __init__.py           # Package initialization
│   ├── layers.py             # Base Layer, Dense layer classes
│   ├── activations.py        # ReLU, Sigmoid, Tanh, Softmax classes
│   ├── losses.py             # MSE loss implementation
│   ├── optimizer.py          # SGD optimizer class
│   └── network.py            # Sequential network class
├── notebooks/                 # Jupyter Notebooks
│   └── project_demo.ipynb    # Complete project demonstration
└── report/
    └── project_report.pdf    # Final PDF report
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd neural-network-library
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Library Components

### Core Architecture
- **Layer**: Abstract base class for all neural network layers
- **Dense**: Fully connected layer with Xavier initialization
- **Sequential**: Network class for composing layers

### Activation Functions
- **ReLU**: Rectified Linear Unit activation
- **Sigmoid**: Sigmoid activation function
- **Tanh**: Hyperbolic tangent activation
- **Softmax**: Softmax activation for multi-class classification

### Training Components
- **MSELoss**: Mean Squared Error loss function
- **SGD**: Stochastic Gradient Descent optimizer

## Applications

### 1. XOR Problem
Validates the library on the classic XOR problem using a 2-4-1 network architecture.

### 2. MNIST Autoencoder
Implements an autoencoder for MNIST digit reconstruction with:
- Encoder: 784 → 256 → 128 → 64 → 32
- Decoder: 32 → 64 → 128 → 256 → 784

### 3. Latent Space SVM Classification
Uses the trained encoder for feature extraction and SVM classification on MNIST digits.

### 4. TensorFlow Baseline Comparison
Compares custom implementation with TensorFlow/Keras for performance and ease of use analysis.

## Usage

See `notebooks/project_demo.ipynb` for complete demonstrations of all library features and applications.

## Features

- **Educational Focus**: Clear, well-documented code for learning neural network fundamentals
- **Modular Design**: Composable components for building custom architectures
- **Gradient Checking**: Numerical validation of backpropagation implementation
- **Progressive Complexity**: From simple XOR to complex autoencoder applications
- **Performance Comparison**: Benchmarking against industry-standard TensorFlow

## Requirements

- Python 3.7+
- NumPy 1.21+
- Matplotlib 3.5+
- Scikit-learn 1.0+
- Jupyter Notebook
- TensorFlow 2.8+ (for baseline comparison)

## License

This project is for educational purposes as part of CSE473 coursework.

## Development Status

🚧 **In Development** - This library is currently being implemented following a structured development plan.

Current implementation status:
- [x] Project structure and core interfaces
- [ ] Base layer architecture
- [ ] Activation functions
- [ ] Loss functions and optimizer
- [ ] Sequential network class
- [ ] Gradient checking validation
- [ ] XOR problem implementation
- [ ] MNIST data handling
- [ ] Autoencoder architecture
- [ ] SVM classification
- [ ] TensorFlow baseline comparison
- [ ] Documentation and demonstration notebook