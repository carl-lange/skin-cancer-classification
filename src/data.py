"""
Data Loading and Preprocessing for Skin Cancer Classification

This module handles loading, preprocessing, and augmentation of skin lesion images
for binary classification between melanoma (MEL) and nevus (NEV) classes.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any


class SkinLesionDataset(Dataset):
    """
    Dataset class for skin lesion images.
    
    Loads images and applies transformations for melanoma vs nevus classification.
    """
    
    def __init__(self, image_dir: str, transform=None, image_size: int = 128):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Directory containing mel_*.png and nev_*.png images
            transform: Optional torchvision transforms
            image_size: Target image size (default 128x128)
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Find all melanoma and nevus images
        self.mel_images = list(self.image_dir.glob("mel_*.png"))
        self.nev_images = list(self.image_dir.glob("nev_*.png"))
        
        # Create dataset with labels (1 for MEL, 0 for NEV)
        self.samples = []
        for img_path in self.mel_images:
            self.samples.append((img_path, 1))  # MEL = 1
        for img_path in self.nev_images:
            self.samples.append((img_path, 0))  # NEV = 0
        
        print(f"Loaded {len(self.mel_images)} MEL and {len(self.nev_images)} NEV images")
        print(f"Total dataset size: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: just convert to tensor and resize
            default_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor()
            ])
            image = default_transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)
    
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset."""
        mel_count = len(self.mel_images)
        nev_count = len(self.nev_images)
        total = mel_count + nev_count
        
        return {
            'MEL': {'count': mel_count, 'percentage': mel_count / total * 100},
            'NEV': {'count': nev_count, 'percentage': nev_count / total * 100},
            'total': total
        }


class DataAugmentation:
    """
    Data augmentation strategies for skin lesion images.
    """
    
    @staticmethod
    def get_default_transforms(image_size: int = 128, normalize: bool = True):
        """
        Get default transforms without augmentation.
        
        Args:
            image_size: Target image size
            normalize: Whether to apply ImageNet normalization
        """
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]
        
        if normalize:
            # ImageNet normalization (commonly used for transfer learning)
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            )
        
        return transforms.Compose(transform_list)
    
    @staticmethod
    def get_augmented_transforms(image_size: int = 128, normalize: bool = True):
        """
        Get augmented transforms for training data.
        
        Includes horizontal flip, rotation, and cropping as per the original assignment.
        
        Args:
            image_size: Target image size
            normalize: Whether to apply ImageNet normalization
        """
        transform_list = [
            transforms.Resize((image_size + 8, image_size + 8)),  # Slightly larger for cropping
            transforms.RandomHorizontalFlip(p=0.5),  # 50% probability
            transforms.RandomRotation(degrees=15),    # ±15 degrees rotation
            transforms.RandomCrop((image_size, image_size)),  # Random crop to target size
            transforms.ToTensor()
        ]
        
        if normalize:
            # Custom normalization based on training data statistics
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            )
        
        return transforms.Compose(transform_list)
    
    @staticmethod
    def get_vgg_transforms(image_size: int = 128):
        """
        Get VGG-16 specific transforms (using ImageNet preprocessing).
        """
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


class SkinLesionDataLoader:
    """
    Data loader manager for skin lesion classification.
    """
    
    def __init__(self, data_dir: str, image_size: int = 128, batch_size: int = 64):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the image files
            image_size: Target image size
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Verify data directory exists and has images
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        self._verify_data()
    
    def _verify_data(self):
        """Verify that the data directory contains the expected image files."""
        data_path = Path(self.data_dir)
        mel_files = list(data_path.glob("mel_*.png"))
        nev_files = list(data_path.glob("nev_*.png"))
        
        if len(mel_files) == 0:
            raise ValueError("No MEL images found. Expected files like 'mel_001.png'")
        if len(nev_files) == 0:
            raise ValueError("No NEV images found. Expected files like 'nev_001.png'")
        
        print(f"Found {len(mel_files)} MEL and {len(nev_files)} NEV images")
    
    def get_datasets(self, augment_training: bool = False, vgg_preprocessing: bool = False):
        """
        Create datasets with appropriate transforms.
        
        Args:
            augment_training: Whether to apply data augmentation
            vgg_preprocessing: Whether to use VGG-16 specific preprocessing
        
        Returns:
            tuple: (dataset, transform_name) for tracking which preprocessing was used
        """
        if vgg_preprocessing:
            transform = DataAugmentation.get_vgg_transforms(self.image_size)
            transform_name = "vgg"
        elif augment_training:
            transform = DataAugmentation.get_augmented_transforms(self.image_size)
            transform_name = "augmented"
        else:
            transform = DataAugmentation.get_default_transforms(self.image_size)
            transform_name = "default"
        
        dataset = SkinLesionDataset(
            image_dir=self.data_dir,
            transform=transform,
            image_size=self.image_size
        )
        
        return dataset, transform_name
    
    def get_data_loader(self, augment_training: bool = False, vgg_preprocessing: bool = False, 
                       shuffle: bool = True, num_workers: int = 0):
        """
        Create a DataLoader for the dataset.
        
        Args:
            augment_training: Whether to apply data augmentation
            vgg_preprocessing: Whether to use VGG-16 preprocessing
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
        
        Returns:
            tuple: (DataLoader, dataset_info)
        """
        dataset, transform_name = self.get_datasets(augment_training, vgg_preprocessing)
        
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Dataset information
        class_dist = dataset.get_class_distribution()
        dataset_info = {
            'total_samples': len(dataset),
            'batch_size': self.batch_size,
            'num_batches': len(data_loader),
            'class_distribution': class_dist,
            'transform_type': transform_name,
            'image_size': self.image_size
        }
        
        return data_loader, dataset_info
    
    def create_train_val_loaders(self, train_split: float = 0.8, augment_training: bool = False,
                                vgg_preprocessing: bool = False, num_workers: int = 0):
        """
        Create training and validation data loaders.
        
        Args:
            train_split: Fraction of data to use for training
            augment_training: Whether to apply augmentation to training data
            vgg_preprocessing: Whether to use VGG preprocessing
            num_workers: Number of worker processes
        
        Returns:
            tuple: (train_loader, val_loader, dataset_info)
        """
        dataset, transform_name = self.get_datasets(
            augment_training=False,  # Don't augment for splitting
            vgg_preprocessing=vgg_preprocessing
        )
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Apply augmentation only to training set if requested
        if augment_training:
            train_transform = DataAugmentation.get_augmented_transforms(self.image_size)
            # Create new dataset with augmented transforms for training
            train_dataset.dataset.transform = train_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Dataset information
        class_dist = dataset.get_class_distribution()
        dataset_info = {
            'total_samples': total_size,
            'train_samples': train_size,
            'val_samples': val_size,
            'batch_size': self.batch_size,
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'class_distribution': class_dist,
            'transform_type': f"{transform_name}_augmented" if augment_training else transform_name,
            'image_size': self.image_size
        }
        
        return train_loader, val_loader, dataset_info


def create_sample_dataloader(data_dir: str = "data/sample", **kwargs):
    """
    Convenience function to create a data loader for the sample data.
    
    Args:
        data_dir: Directory containing sample images
        **kwargs: Additional arguments for SkinLesionDataLoader
    
    Returns:
        SkinLesionDataLoader instance
    """
    return SkinLesionDataLoader(data_dir, **kwargs)


def print_dataset_info(dataset_info: Dict[str, Any]):
    """
    Print formatted dataset information.
    
    Args:
        dataset_info: Dictionary containing dataset statistics
    """
    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)
    
    print(f"Total samples: {dataset_info['total_samples']}")
    if 'train_samples' in dataset_info:
        print(f"Training samples: {dataset_info['train_samples']}")
        print(f"Validation samples: {dataset_info['val_samples']}")
    
    print(f"Batch size: {dataset_info['batch_size']}")
    print(f"Image size: {dataset_info['image_size']}x{dataset_info['image_size']}")
    print(f"Transform type: {dataset_info['transform_type']}")
    
    print("\nClass Distribution:")
    class_dist = dataset_info['class_distribution']
    for class_name, info in class_dist.items():
        if class_name != 'total':
            print(f"  {class_name}: {info['count']} samples ({info['percentage']:.1f}%)")
    
    print("=" * 50)


def verify_sample_data(data_dir: str = "data/sample"):
    """
    Verify that the sample data is properly set up.
    
    Args:
        data_dir: Directory to check for sample images
    """
    try:
        loader = create_sample_dataloader(data_dir)
        data_loader, info = loader.get_data_loader()
        print_dataset_info(info)
        
        # Test loading a batch
        sample_batch = next(iter(data_loader))
        images, labels = sample_batch
        print(f"\nSample batch shape: {images.shape}")
        print(f"Sample labels: {labels}")
        print("✓ Sample data verification successful!")
        
        return True
    except Exception as e:
        print(f"✗ Sample data verification failed: {e}")
        return False