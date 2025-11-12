"""
Dataset and DataLoader for FaceNet-based Face Deepfake Detection
Includes face detection, alignment, and face-specific augmentations
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
from facenet_pytorch import MTCNN
import config_facenet as config


class FaceDataset(Dataset):
    """
    Dataset for face images with optional MTCNN face detection
    """
    
    def __init__(self, data_dir, transform=None, use_face_detection=True):
        """
        Args:
            data_dir: Directory with 'real' and 'fake' subdirectories
            transform: torchvision transforms
            use_face_detection: Whether to detect and crop faces
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_face_detection = use_face_detection
        
        # Initialize MTCNN for face detection
        # Use CPU for MTCNN to avoid multiprocessing issues with MPS
        if use_face_detection:
            self.mtcnn = MTCNN(
                image_size=config.IMG_SIZE,
                margin=config.FACE_DETECTION_MARGIN,
                min_face_size=20,
                thresholds=[0.6, 0.7, config.FACE_DETECTION_THRESHOLD],
                keep_all=False,  # Keep only the most prominent face
                device='cpu'  # Always use CPU for MTCNN in dataset
            )
        else:
            self.mtcnn = None
        
        # Load image paths and labels
        self.samples = []
        self._load_samples()
        
        print(f"  Loaded {len(self.samples)} images from {data_dir.name}")
        if self.use_face_detection:
            print(f"  Using MTCNN face detection (margin={config.FACE_DETECTION_MARGIN})")
    
    def _load_samples(self):
        """Load all image paths and labels"""
        for class_name in ['real', 'fake']:
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            label = config.CLASS_TO_IDX[class_name]
            
            # Support multiple image formats
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.jfif', '*.webp']:
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), label))
    
    def __len__(self):
        return len(self.samples)
    
    def _detect_and_crop_face(self, image):
        """
        Detect face using MTCNN and crop/align it
        Falls back to center crop if no face detected
        """
        try:
            # MTCNN expects PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Detect and crop face
            face = self.mtcnn(image)
            
            if face is not None:
                # Convert tensor to PIL Image
                # MTCNN returns normalized tensor, denormalize it
                face = face.permute(1, 2, 0).cpu().numpy()
                face = ((face * 128) + 128).clip(0, 255).astype(np.uint8)
                face = Image.fromarray(face)
                return face
            else:
                # No face detected - use center crop
                return self._center_crop(image)
                
        except Exception as e:
            # Fall back to center crop on error
            return self._center_crop(image)
    
    def _center_crop(self, image):
        """Center crop image to square"""
        width, height = image.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        return image.crop((left, top, right, bottom))
    
    def __getitem__(self, idx):
        """Get image and label"""
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Detect and crop face if enabled
            if self.use_face_detection and self.mtcnn is not None:
                image = self._detect_and_crop_face(image)
            else:
                # Just resize without face detection
                image = image.resize((config.IMG_SIZE, config.IMG_SIZE))
            
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
    Get transforms for FaceNet
    FaceNet expects images normalized with mean and std
    """
    if is_training:
        transform_list = [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            
            # Face-specific augmentations
            transforms.RandomHorizontalFlip(
                p=config.AUGMENTATION_CONFIG['horizontal_flip_prob']
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
            
            # Convert to tensor
            transforms.ToTensor(),
            
            # Random Erasing (simulates occlusions)
            transforms.RandomErasing(
                p=config.AUGMENTATION_CONFIG['random_erasing_prob'],
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3)
            ),
            
            # FaceNet normalization
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ]
    else:
        # Validation/Test transforms (no augmentation)
        transform_list = [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ]
    
    return transforms.Compose(transform_list)


def create_facenet_dataloaders():
    """
    Create train, validation, and test dataloaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"\n{'='*60}")
    print("CREATING FACENET DATALOADERS")
    print(f"{'='*60}\n")
    
    # Get transforms
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = FaceDataset(
        config.TRAIN_DIR,
        transform=train_transform,
        use_face_detection=config.USE_FACE_DETECTION
    )
    
    val_dataset = FaceDataset(
        config.VAL_DIR,
        transform=val_transform,
        use_face_detection=config.USE_FACE_DETECTION
    )
    
    test_dataset = FaceDataset(
        config.TEST_DIR,
        transform=val_transform,
        use_face_detection=config.USE_FACE_DETECTION
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False
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
    print("Testing FaceNet dataset creation...\n")
    
    train_loader, val_loader, test_loader = create_facenet_dataloaders()
    
    # Test loading a batch
    print("\nTesting batch loading...")
    for images, labels in train_loader:
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Labels: {labels[:5].tolist()}")
        break
    
    print("\n FaceNet dataset working correctly!")
