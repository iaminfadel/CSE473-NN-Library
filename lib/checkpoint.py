"""
Neural network checkpointing system with .amin file format.

This module provides functionality to save and load neural network training
checkpoints using a custom .amin (Amin's Neural Network) file format.
"""

import pickle
import json
import numpy as np
import os
from datetime import datetime
from .network import Sequential
from .layers import Dense
from .optimizer import SGD


class NetworkCheckpoint:
    """
    Handles saving and loading neural network checkpoints.
    
    The .amin file format stores:
    - Network architecture (layer types, sizes, activation functions)
    - All network parameters (weights, biases)
    - Optimizer state and configuration
    - Training metadata (epoch, loss history, etc.)
    """
    
    def __init__(self):
        """Initialize checkpoint handler."""
        self.version = "1.0"
    
    def save_checkpoint(self, network, optimizer, epoch, loss_history, 
                       filepath, metadata=None):
        """
        Save a complete training checkpoint to .amin file.
        
        Args:
            network: Sequential network instance
            optimizer: Optimizer instance (SGD, etc.)
            epoch: Current training epoch
            loss_history: List of loss values from training
            filepath: Path to save checkpoint (should end with .amin)
            metadata: Optional dictionary with additional metadata
            
        Returns:
            str: Path to saved checkpoint file
        """
        # Ensure filepath has .amin extension
        if not filepath.endswith('.amin'):
            filepath += '.amin'
        
        # Create checkpoint data structure
        checkpoint_data = {
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'loss_history': loss_history,
            'network_architecture': self._extract_network_architecture(network),
            'network_parameters': self._extract_network_parameters(network),
            'optimizer_config': self._extract_optimizer_config(optimizer),
            'metadata': metadata or {}
        }
        
        # Save to file using pickle for binary efficiency
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Checkpoint saved successfully to: {filepath}")
            print(f"Epoch: {epoch}, Latest Loss: {loss_history[-1] if loss_history else 'N/A'}")
            return filepath
            
        except Exception as e:
            raise RuntimeError(f"Failed to save checkpoint: {str(e)}")
    
    def load_checkpoint(self, filepath):
        """
        Load a training checkpoint from .amin file.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            dict: Dictionary containing:
                - 'network': Reconstructed Sequential network
                - 'optimizer': Reconstructed optimizer
                - 'epoch': Training epoch
                - 'loss_history': Training loss history
                - 'metadata': Additional metadata
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Validate checkpoint version
            if checkpoint_data.get('version') != self.version:
                print(f"Warning: Checkpoint version {checkpoint_data.get('version')} "
                      f"may not be compatible with current version {self.version}")
            
            # Reconstruct network
            network = self._reconstruct_network(
                checkpoint_data['network_architecture'],
                checkpoint_data['network_parameters']
            )
            
            # Reconstruct optimizer
            optimizer = self._reconstruct_optimizer(checkpoint_data['optimizer_config'])
            
            print(f"Checkpoint loaded successfully from: {filepath}")
            print(f"Epoch: {checkpoint_data['epoch']}, "
                  f"Loss History Length: {len(checkpoint_data['loss_history'])}")
            
            return {
                'network': network,
                'optimizer': optimizer,
                'epoch': checkpoint_data['epoch'],
                'loss_history': checkpoint_data['loss_history'],
                'metadata': checkpoint_data.get('metadata', {}),
                'timestamp': checkpoint_data.get('timestamp')
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
    
    def _extract_network_architecture(self, network):
        """Extract network architecture information."""
        architecture = {
            'type': 'Sequential',
            'layers': []
        }
        
        for i, layer in enumerate(network.layers):
            layer_info = {
                'index': i,
                'type': layer.__class__.__name__
            }
            
            if isinstance(layer, Dense):
                layer_info.update({
                    'input_size': layer.input_size,
                    'output_size': layer.output_size
                })
            
            # Add more layer types as needed
            architecture['layers'].append(layer_info)
        
        return architecture
    
    def _extract_network_parameters(self, network):
        """Extract all network parameters."""
        parameters = {}
        
        for i, layer in enumerate(network.layers):
            layer_params = {}
            
            if isinstance(layer, Dense):
                layer_params = {
                    'weights': layer.weights.copy(),
                    'biases': layer.biases.copy()
                }
            
            parameters[f'layer_{i}'] = layer_params
        
        return parameters
    
    def _extract_optimizer_config(self, optimizer):
        """Extract optimizer configuration and state."""
        config = {
            'type': optimizer.__class__.__name__
        }
        
        if isinstance(optimizer, SGD):
            config.update({
                'learning_rate': optimizer.learning_rate,
                'weight_decay': optimizer.weight_decay
            })
        
        # Add more optimizer types as needed
        return config
    
    def _reconstruct_network(self, architecture, parameters):
        """Reconstruct network from saved architecture and parameters."""
        if architecture['type'] != 'Sequential':
            raise ValueError(f"Unsupported network type: {architecture['type']}")
        
        network = Sequential()
        
        for layer_info in architecture['layers']:
            layer_type = layer_info['type']
            
            if layer_type == 'Dense':
                layer = Dense(
                    input_size=layer_info['input_size'],
                    output_size=layer_info['output_size']
                )
                
                # Load saved parameters
                layer_params = parameters[f"layer_{layer_info['index']}"]
                layer.weights = layer_params['weights'].copy()
                layer.biases = layer_params['biases'].copy()
                
                network.add(layer)
            
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
        
        return network
    
    def _reconstruct_optimizer(self, config):
        """Reconstruct optimizer from saved configuration."""
        optimizer_type = config['type']
        
        if optimizer_type == 'SGD':
            return SGD(
                learning_rate=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def list_checkpoints(self, directory='.'):
        """
        List all .amin checkpoint files in a directory.
        
        Args:
            directory: Directory to search for checkpoints
            
        Returns:
            list: List of checkpoint file paths
        """
        checkpoints = []
        
        for filename in os.listdir(directory):
            if filename.endswith('.amin'):
                filepath = os.path.join(directory, filename)
                checkpoints.append(filepath)
        
        return sorted(checkpoints)
    
    def get_checkpoint_info(self, filepath):
        """
        Get basic information about a checkpoint without fully loading it.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            dict: Basic checkpoint information
        """
        try:
            with open(filepath, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            return {
                'filepath': filepath,
                'version': checkpoint_data.get('version'),
                'timestamp': checkpoint_data.get('timestamp'),
                'epoch': checkpoint_data.get('epoch'),
                'num_layers': len(checkpoint_data.get('network_architecture', {}).get('layers', [])),
                'loss_history_length': len(checkpoint_data.get('loss_history', [])),
                'final_loss': checkpoint_data.get('loss_history', [])[-1] if checkpoint_data.get('loss_history') else None,
                'optimizer_type': checkpoint_data.get('optimizer_config', {}).get('type'),
                'metadata': checkpoint_data.get('metadata', {})
            }
            
        except Exception as e:
            return {'filepath': filepath, 'error': str(e)}


def save_checkpoint(network, optimizer, epoch, loss_history, filepath, metadata=None):
    """
    Convenience function to save a checkpoint.
    
    Args:
        network: Sequential network instance
        optimizer: Optimizer instance
        epoch: Current training epoch
        loss_history: List of loss values
        filepath: Path to save checkpoint
        metadata: Optional metadata dictionary
        
    Returns:
        str: Path to saved checkpoint file
    """
    checkpoint_handler = NetworkCheckpoint()
    return checkpoint_handler.save_checkpoint(
        network, optimizer, epoch, loss_history, filepath, metadata
    )


def load_checkpoint(filepath):
    """
    Convenience function to load a checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        
    Returns:
        dict: Loaded checkpoint data
    """
    checkpoint_handler = NetworkCheckpoint()
    return checkpoint_handler.load_checkpoint(filepath)


def list_checkpoints(directory='.'):
    """
    Convenience function to list checkpoints in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        list: List of checkpoint file paths
    """
    checkpoint_handler = NetworkCheckpoint()
    return checkpoint_handler.list_checkpoints(directory)