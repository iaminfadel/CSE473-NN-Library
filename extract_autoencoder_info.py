import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the final autoencoder results
with open('autoencoder_results_final.pkl', 'rb') as f:
    results = pickle.load(f)

print('=== AUTOENCODER RESULTS SUMMARY ===')
print()
print('Architecture:')
print(f'  Input dimension: 784 (28x28 MNIST images)')
print(f'  Latent dimension: {results["autoencoder"].latent_dim}')
print(f'  Compression ratio: {784 / results["autoencoder"].latent_dim:.1f}x')
print()
print('Dataset:')
print(f'  Training samples: {results["data"]["X_train"].shape[0]:,}')
print(f'  Validation samples: {results["data"]["X_val"].shape[0]:,}')
print(f'  Test samples: {results["data"]["X_test"].shape[0]:,}')
print()
print('Training Results:')
print(f'  Epochs trained: {len(results["history"]["train_losses"])}')
print(f'  Final training loss: {results["history"]["train_losses"][-1]:.6f}')
print(f'  Final validation loss: {results["history"]["val_losses"][-1]:.6f}')
print(f'  Best validation loss: {min(results["history"]["val_losses"]):.6f}')
print()
print('Test Metrics:')
print(f'  MSE: {results["test_metrics"]["mse"]:.6f}')
print(f'  MAE: {results["test_metrics"]["mae"]:.6f}')
print(f'  BCE Loss: {results["test_metrics"]["bce_loss"]:.6f}')
print(f'  Mean per-sample MSE: {results["test_metrics"]["mean_per_sample_mse"]:.6f}')
print(f'  Std per-sample MSE: {results["test_metrics"]["std_per_sample_mse"]:.6f}')

# Create loss curve plot
plt.figure(figsize=(10, 6))
train_losses = results["history"]["train_losses"]
val_losses = results["history"]["val_losses"]
epochs = range(1, len(train_losses) + 1)

print(f'Training losses length: {len(train_losses)}')
print(f'Validation losses length: {len(val_losses)}')

# Ensure both have the same length
min_length = min(len(train_losses), len(val_losses))
epochs = range(1, min_length + 1)
train_losses = train_losses[:min_length]
val_losses = val_losses[:min_length]

plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Autoencoder Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/autoencoder_loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Create reconstruction examples
autoencoder = results["autoencoder"]
X_test = results["data"]["X_test"]
y_test = results["data"]["y_test"]

# Select a few test samples for visualization
np.random.seed(42)
n_examples = 10
indices = np.random.choice(len(X_test), n_examples, replace=False)
test_samples = X_test[indices]
test_labels = y_test[indices]

# Get reconstructions
reconstructions = autoencoder.forward(test_samples)

# Create visualization
fig, axes = plt.subplots(2, n_examples, figsize=(15, 4))
for i in range(n_examples):
    # Original images
    axes[0, i].imshow(test_samples[i].reshape(28, 28), cmap='gray')
    axes[0, i].set_title(f'Original\n(Label: {test_labels[i]})')
    axes[0, i].axis('off')
    
    # Reconstructed images
    axes[1, i].imshow(reconstructions[i].reshape(28, 28), cmap='gray')
    mse = np.mean((test_samples[i] - reconstructions[i])**2)
    axes[1, i].set_title(f'Reconstructed\n(MSE: {mse:.4f})')
    axes[1, i].axis('off')

plt.suptitle('Autoencoder Reconstruction Examples', fontsize=16)
plt.tight_layout()
plt.savefig('report/autoencoder_reconstructions.png', dpi=300, bbox_inches='tight')
plt.close()

# Create latent space visualization (2D projection)
from sklearn.decomposition import PCA

# Extract latent features for a subset of test data
n_viz = 2000
viz_indices = np.random.choice(len(X_test), n_viz, replace=False)
X_viz = X_test[viz_indices]
y_viz = y_test[viz_indices]

# Get latent representations
encoder = results["encoder"]
latent_features = encoder.forward(X_viz)

# Apply PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_features)

# Create scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y_viz, cmap='tab10', alpha=0.6, s=20)
plt.colorbar(scatter, label='Digit Class')
plt.xlabel(f'First Principal Component (explains {pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Second Principal Component (explains {pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Latent Space Visualization (PCA Projection)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/latent_space_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print()
print('Visualizations saved:')
print('  - report/autoencoder_loss_curve.png')
print('  - report/autoencoder_reconstructions.png')
print('  - report/latent_space_visualization.png')