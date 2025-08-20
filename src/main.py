#!/usr/bin/env python3
"""
Skin Cancer Classification System

A command-line interface for training and evaluating CNN models
for skin cancer classification. Supports multiple CNN architectures,
data augmentation, and comprehensive evaluation metrics.

Usage:
    python main.py --model baseline --mode train --data-dir data/sample
    python main.py --model resnet --mode train --data-dir data/sample --augment
    python main.py --model vgg16 --mode eval --model-path models/vgg16_model.pth
    python main.py --model batch_norm --mode demo --data-dir data/sample
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models import get_model, count_parameters, model_summary
from data import SkinLesionDataLoader, print_dataset_info, verify_sample_data


class SkinCancerTrainer:
    """Main trainer class for skin cancer classification models."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            model_name: Name of the CNN model to use
            config: Configuration dictionary with hyperparameters
        """
        self.model_name = model_name
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Training hyperparameters
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.batch_size = config.get('batch_size', 64)
        self.num_epochs = config.get('num_epochs', 30)
        self.patience = config.get('patience', 10)
        self.image_size = config.get('image_size', 128)
        
        # Data augmentation and preprocessing options
        self.augment_training = config.get('augment_training', False)
        self.vgg_preprocessing = config.get('vgg_preprocessing', False)
        
        print(f"Trainer initialized for {model_name} on {self.device}")
        
        # Initialize model
        self._setup_model()
    
    def _setup_model(self):
        """Setup the CNN model based on the specified architecture."""
        try:
            if self.model_name == 'vgg16':
                self.model = get_model(self.model_name, pretrained=True)
                self.vgg_preprocessing = True  # Force VGG preprocessing for VGG models
            else:
                self.model = get_model(self.model_name)
            
            self.model = self.model.to(self.device)
            
            # Print model summary
            print(f"\\nModel: {self.model_name}")
            print(f"Parameters: {count_parameters(self.model):,}")
            print(f"Device: {self.device}")
            
        except Exception as e:
            print(f"Error setting up model {self.model_name}: {e}")
            sys.exit(1)
    
    def load_data(self, data_dir: str = "data/sample") -> Tuple[DataLoader, DataLoader, Dict]:
        """
        Load and preprocess the skin lesion data.
        
        Args:
            data_dir: Directory containing the image files
            
        Returns:
            Tuple of (train_loader, val_loader, dataset_info)
        """
        try:
            # Create data loader
            skin_loader = SkinLesionDataLoader(
                data_dir=data_dir,
                image_size=self.image_size,
                batch_size=self.batch_size
            )
            
            # Create train/validation split
            train_loader, val_loader, dataset_info = skin_loader.create_train_val_loaders(
                train_split=0.8,
                augment_training=self.augment_training,
                vgg_preprocessing=self.vgg_preprocessing,
                num_workers=0  # Set to 0 to avoid multiprocessing issues
            )
            
            print_dataset_info(dataset_info)
            
            return train_loader, val_loader, dataset_info
            
        except Exception as e:
            print(f"Error loading data from {data_dir}: {e}")
            print("Make sure you have sample images in the data directory.")
            sys.exit(1)
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the CNN model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Optional path to save the trained model
            
        Returns:
            Dictionary with training history and final metrics
        """
        print(f"\\nStarting training for {self.num_epochs} epochs...")
        
        # Setup loss function and optimizer
        criterion = nn.BCELoss()  # Binary cross-entropy for binary classification
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Early stopping setup
        best_val_accuracy = 0.0
        patience_counter = 0
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'epochs': []
        }
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['epochs'].append(epoch + 1)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1:2d}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Early stopping check
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                
                # Save best model
                if save_path:
                    self._save_model(save_path, epoch, val_acc)
                    
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"\\nEarly stopping at epoch {epoch+1} (patience={self.patience})")
                break
        
        total_time = time.time() - start_time
        
        print(f"\\nTraining completed!")
        print(f"Total time: {total_time:.1f}s")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        
        return {
            'history': history,
            'best_val_accuracy': best_val_accuracy,
            'total_epochs': epoch + 1,
            'total_time': total_time
        }
    
    def _train_epoch(self, train_loader: DataLoader, criterion, optimizer) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images).squeeze()
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate_model(self, data_loader: DataLoader, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the model on given data.
        
        Args:
            data_loader: Data loader for evaluation
            model_path: Optional path to load model from
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_path:
            self._load_model(model_path)
        
        print(f"\\nEvaluating {self.model_name} model...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                outputs = self.model(images).squeeze()
                predicted = (outputs > 0.5).float().cpu()
                
                all_predictions.extend(predicted.numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Convert to class names for better reporting
        label_names = ['NEV', 'MEL']
        class_report = classification_report(
            all_labels, all_predictions, 
            target_names=label_names, 
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'model_name': self.model_name
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=label_names))
        
        return results
    
    def demo(self, data_loader: DataLoader, model_path: Optional[str] = None, num_samples: int = 5):
        """
        Run interactive demonstration of the trained model.
        
        Args:
            data_loader: Data loader containing images to demo
            model_path: Optional path to load model from
            num_samples: Number of sample predictions to show
        """
        if model_path:
            self._load_model(model_path)
        
        print(f"\\nRunning model demonstration with {num_samples} samples...")
        
        self.model.eval()
        
        # Get a batch of data
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        
        # Limit to requested number of samples
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        # Make predictions
        with torch.no_grad():
            images_device = images.to(self.device)
            outputs = self.model(images_device).squeeze()
            probabilities = outputs.cpu().numpy()
            predictions = (outputs > 0.5).float().cpu().numpy()
        
        print("\\nSample Predictions:")
        print("=" * 60)
        
        class_names = ['NEV (Nevus)', 'MEL (Melanoma)']
        
        for i in range(len(images)):
            true_label = int(labels[i].item())
            pred_label = int(predictions[i])
            confidence = probabilities[i]
            
            print(f"Sample {i+1}:")
            print(f"  True Label: {class_names[true_label]}")
            print(f"  Predicted: {class_names[pred_label]}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Correct: {'✓' if pred_label == true_label else '✗'}")
            print()
    
    def _save_model(self, save_path: str, epoch: int, val_acc: float):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'epoch': epoch,
            'val_accuracy': val_acc,
            'config': self.config
        }, save_path)
    
    def _load_model(self, model_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")


def create_training_plots(history: Dict[str, Any], save_path: Optional[str] = None):
    """
    Create training plots from history.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = history['epochs']
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training plots saved to {save_path}")
    else:
        plt.show()


def get_default_config(model_name: str) -> Dict[str, Any]:
    """Get default configuration for each model."""
    base_config = {
        'learning_rate': 0.0001,
        'batch_size': 64,
        'num_epochs': 30,
        'patience': 10,
        'image_size': 128,
        'augment_training': False,
        'vgg_preprocessing': False
    }
    
    # Model-specific configurations
    if model_name == 'vgg16':
        base_config.update({
            'learning_rate': 0.0001,  # Lower learning rate for transfer learning
            'vgg_preprocessing': True,
            'batch_size': 32  # Smaller batch size for larger model
        })
    elif model_name in ['resnet']:
        base_config.update({
            'learning_rate': 0.001,  # Higher learning rate for ResNet
            'augment_training': True  # ResNet benefits from augmentation
        })
    elif 'norm' in model_name:
        base_config.update({
            'batch_size': 32,  # Smaller batch for normalization models
            'learning_rate': 0.001
        })
    
    return base_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate CNN models for skin cancer classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--model', '-m',
                        choices=['baseline', 'batch_norm', 'group_norm', 'layer_norm', 
                                'resnet', 'vgg16'],
                        default='baseline',
                        help='CNN model architecture to use')
    parser.add_argument('--mode',
                        choices=['train', 'eval', 'demo', 'verify'],
                        required=True,
                        help='Mode: train, evaluate, demo, or verify data')
    parser.add_argument('--data-dir', '-d', 
                        default='data/sample',
                        help='Directory containing image files')
    parser.add_argument('--model-path', '-p',
                        help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation for training')
    parser.add_argument('--plot-path',
                        help='Path to save training plots')
    parser.add_argument('--demo-samples', type=int, default=5,
                        help='Number of samples to show in demo mode')
    
    args = parser.parse_args()
    
    # Special case: just verify sample data
    if args.mode == 'verify':
        print("Verifying sample data...")
        if verify_sample_data(args.data_dir):
            print("✓ Sample data verification successful!")
        else:
            print("✗ Sample data verification failed!")
        return
    
    # Get configuration
    config = get_default_config(args.model)
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['augment_training'] = args.augment
    
    # Create trainer
    trainer = SkinCancerTrainer(args.model, config)
    
    try:
        if args.mode == 'train':
            # Load data
            train_loader, val_loader, dataset_info = trainer.load_data(args.data_dir)
            
            # Set default model path if not provided
            model_path = args.model_path or f"models/{args.model}_model.pth"
            
            # Train model
            results = trainer.train_model(train_loader, val_loader, model_path)
            
            print(f"\\nTraining completed!")
            print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
            
            # Create training plots
            if args.plot_path:
                create_training_plots(results['history'], args.plot_path)
            
        elif args.mode == 'eval':
            # Load data for evaluation
            data_loader, _, dataset_info = trainer.load_data(args.data_dir)
            
            # Evaluate model
            results = trainer.evaluate_model(data_loader, args.model_path)
            
        elif args.mode == 'demo':
            # Load data for demo
            data_loader, _, dataset_info = trainer.load_data(args.data_dir)
            
            # Run demo
            trainer.demo(data_loader, args.model_path, args.demo_samples)
    
    except KeyboardInterrupt:
        print("\\nInterrupted by user")
    except Exception as e:
        print(f"\\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()