"""
Dataset loader for Buildings Deepfake Detection
Handles loading and splitting the Buildings dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import config_buildings as config


class BuildingsDataset(Dataset):
    """
    Dataset for building images (real and fake)
    """
    
    def __init__(self, transform=None):
        """
        Args:
            transform: torchvision transforms
        """
        self.transform = transform
        self.samples = []
        self._load_samples()
        
        print(f"  Loaded {len(self.samples)} images")
    
    def _load_samples(self):
        """Load all image paths and labels"""
        # Load Real buildings
        real_dir = config.REAL_DIR
        if real_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.jfif']:
                for img_path in real_dir.glob(ext):
                    self.samples.append((str(img_path), 0))  # 0 = Real
        
        # Load Fake buildings
        fake_dir = config.FAKE_DIR
        if fake_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.jfif']:
                for img_path in fake_dir.glob(ext):
                    self.samples.append((str(img_path), 1))  # 1 = Fake
        
        # Shuffle samples
        np.random.seed(config.RANDOM_SEED)
        np.random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get image and label"""
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image on error
            blank_image = torch.zeros(3, config.IMG_SIZE, config.IMG_SIZE)
            return blank_image, label


def get_transforms(is_training=True):
    """
    Get transforms for buildings
    Heavy augmentation since dataset is small
    """
    if is_training:
        transform_list = [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            
            # Heavy augmentation for small dataset
            transforms.RandomHorizontalFlip(
                p=config.AUGMENTATION_CONFIG['horizontal_flip_prob']
            ),
            transforms.RandomVerticalFlip(
                p=config.AUGMENTATION_CONFIG['vertical_flip_prob']
            ),
            transforms.RandomRotation(
                degrees=config.AUGMENTATION_CONFIG['rotation_degrees']
            ),
            transforms.ColorJitter(
                brightness=config.AUGMENTATION_CONFIG['color_jitter']['brightness'],
                contrast=config.AUGMENTATION_CONFIG['color_jitter']['contrast'],
                saturation=config.AUGMENTATION_CONFIG['color_jitter']['saturation'],
                hue=config.AUGMENTATION_CONFIG['color_jitter']['hue']
            ),
            transforms.RandomAffine(
                degrees=config.AUGMENTATION_CONFIG['random_affine']['degrees'],
                translate=config.AUGMENTATION_CONFIG['random_affine']['translate'],
                scale=config.AUGMENTATION_CONFIG['random_affine']['scale']
            ),
            transforms.RandomPerspective(
                distortion_scale=0.3,
                p=config.AUGMENTATION_CONFIG['random_perspective_prob']
            ),
            
            # Convert to tensor
            transforms.ToTensor(),
            
            # Additional augmentations
            transforms.RandomErasing(
                p=config.AUGMENTATION_CONFIG['random_erasing_prob'],
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3)
            ),
            
            # Normalize
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    else:
        # Validation/Test transforms (no augmentation)
        transform_list = [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    
    return transforms.Compose(transform_list)


def create_buildings_dataloaders():
    """
    Create train, validation, and test dataloaders
    Splits the dataset automatically
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"\n{'='*60}")
    print("CREATING BUILDINGS DATALOADERS")
    print(f"{'='*60}\n")
    
    print("Loading full dataset...")
    # Load full dataset
    full_dataset = BuildingsDataset(transform=None)
    
    total_size = len(full_dataset)
    train_size = int(config.TRAIN_RATIO * total_size)
    val_size = int(config.VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nSplitting dataset:")
    print(f"  Total: {total_size} images")
    print(f"  Train: {train_size} images ({config.TRAIN_RATIO:.0%})")
    print(f"  Val:   {val_size} images ({config.VAL_RATIO:.0%})")
    print(f"  Test:  {test_size} images ({config.TEST_RATIO:.0%})")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    # Apply transforms to each split
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Wrap with transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"\nDataloader Summary:")
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Workers: {config.NUM_WORKERS}")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset creation
    print("Testing Buildings dataset creation...\n")
    
    train_loader, val_loader, test_loader = create_buildings_dataloaders()
    
    # Test loading a batch
    print("\nTesting batch loading...")
    for images, labels in train_loader:
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Labels: {labels[:5].tolist()}")
        break
    
    print("\n✅ Buildings dataset working correctly!")
