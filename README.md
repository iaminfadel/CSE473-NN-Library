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

### 1. XOR Problem ✅ **COMPLETED**
Validates the library on the classic XOR problem using a 2-4-1 network architecture.

**Results:**
- Network architecture: 2 inputs → 4 hidden (Tanh) → 1 output (Sigmoid)
- Training: 5000 epochs, learning rate 0.5, MSE loss
- Final accuracy: 100% on all 4 XOR inputs
- Final loss: 0.0003 (converged from initial 0.25)
- All gradient checks pass with relative errors < 10⁻⁸

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

### Milestone 1: ✅ **COMPLETED**
Core neural network library implementation and XOR problem validation.

**Completed Features:**
- [x] Project structure and core interfaces
- [x] Base layer architecture (Layer, Dense)
- [x] Activation functions (ReLU, Sigmoid, Tanh, Softmax)
- [x] Loss functions (MSE) and optimizer (SGD)
- [x] Sequential network class
- [x] Gradient checking validation
- [x] XOR problem implementation and validation
- [x] Comprehensive documentation and report

**Milestone 1 Results:**
- ✅ All gradient checks pass with errors < 10⁻⁵
- ✅ XOR problem solved with 100% accuracy
- ✅ Network achieves final loss of ~0.0003 after 5000 epochs
- ✅ Complete technical report with mathematical analysis

### Upcoming Milestones:
- [ ] **Milestone 2**: MNIST autoencoder implementation
- [ ] **Milestone 3**: SVM classification with latent features
- [ ] **Milestone 4**: TensorFlow baseline comparison