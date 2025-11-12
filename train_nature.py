"""
Training script specifically for Nature dataset
Automatically splits data into 75:15:10 (train:val:test) ratio
Model will be saved as 'nature' in output/models/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import os

# ===============================
# CONFIGURATION
# ===============================
class NatureConfig:
    """Configuration for Nature dataset training"""
    # Paths
    BASE_DIR = Path("/Users/anmol/Documents/Data")
    NATURE_DIR = BASE_DIR / "datasets" / "Nature"
    OUTPUT_DIR = BASE_DIR / "output"
    MODEL_DIR = OUTPUT_DIR / "models"
    LOGS_DIR = OUTPUT_DIR / "logs"
    RESULTS_DIR = OUTPUT_DIR / "results"
    
    # Model settings
    MODEL_NAME = "efficientnet_b3"
    PRETRAINED = True
    NUM_CLASSES = 1  # Binary classification
    
    # Training settings
    IMG_SIZE = 224
    BATCH_SIZE = 32 
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Data split (75:15:10 = train:val:test)
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.10
    
    # Data loading
    NUM_WORKERS = 6
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    
    # Training options
    USE_MIXED_PRECISION = True
    GRADIENT_CLIPPING = 1.0
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Device - Force CUDA check
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device('cpu')
    
    # Logging
    LOG_INTERVAL = 5
    RANDOM_SEED = 42
    
    # Model save name
    MODEL_SAVE_NAME = "nature"

config = NatureConfig()

# Create output directories
for dir_path in [config.OUTPUT_DIR, config.MODEL_DIR, config.LOGS_DIR, config.RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set random seeds
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.RANDOM_SEED)

# ===============================
# DATASET CLASS
# ===============================
class NatureDataset(Dataset):
    """Dataset class for Nature images (fake and real)"""
    
    def __init__(self, nature_dir, transform=None):
        self.nature_dir = Path(nature_dir)
        self.transform = transform
        self.samples = []
        
        # Load fake images (label = 1)
        fake_dir = self.nature_dir / "fake"
        if fake_dir.exists():
            for img_path in fake_dir.glob("*"):
                # Skip Mac resource fork files
                if img_path.name.startswith("._") or img_path.name.startswith("."):
                    continue
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.jfif']:
                    self.samples.append((str(img_path), 1))  # 1 = Fake
        
        # Load real images (label = 0)
        real_dir = self.nature_dir / "real"
        if real_dir.exists():
            for img_path in real_dir.glob("*"):
                # Skip Mac resource fork files
                if img_path.name.startswith("._") or img_path.name.startswith("."):
                    continue
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.jfif']:
                    self.samples.append((str(img_path), 0))  # 0 = Real
        
        print(f"Loaded {len(self.samples)} images from Nature dataset")
        fake_count = sum(1 for _, label in self.samples if label == 1)
        real_count = sum(1 for _, label in self.samples if label == 0)
        print(f"  Real: {real_count}, Fake: {fake_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.float32)
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image in case of error
            if self.transform:
                image = self.transform(Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE)))
            else:
                image = torch.zeros(3, config.IMG_SIZE, config.IMG_SIZE)
            return image, torch.tensor(label, dtype=torch.float32)

# ===============================
# DATA TRANSFORMS
# ===============================
def get_transforms(train=True):
    """Get image transforms for training or validation"""
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ===============================
# MODEL
# ===============================
def create_model():
    """Create EfficientNet-B3 model"""
    print(f"Creating {config.MODEL_NAME} model...")
    
    if config.MODEL_NAME == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=config.PRETRAINED)
        # Modify final classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, config.NUM_CLASSES)
        )
    else:
        raise ValueError(f"Model {config.MODEL_NAME} not supported")
    
    return model.to(config.DEVICE)

# ===============================
# TRAINING FUNCTION
# ===============================
def train_epoch(model, train_loader, criterion, optimizer, scaler, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [Train]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Mixed precision training: use scaler if available
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIPPING)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIPPING)
            optimizer.step()
        
        # Calculate accuracy
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()
        
        # Update progress bar
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

# ===============================
# VALIDATION FUNCTION
# ===============================
def validate(model, val_loader, criterion, epoch, phase="Val"):
    """Validate the model"""
    # Added scaler parameter for mixed precision
    def validate(model, val_loader, criterion, epoch, phase="Val", scaler=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [{phase}]")
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE).unsqueeze(1)
            
            # Use AMP only if scaler is available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Calculate accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
            
            # Update progress bar
            avg_loss = running_loss / (pbar.n + 1)
            accuracy = 100.0 * correct / total
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

# ===============================
# MAIN TRAINING
# ===============================
def main():
    print("="*60)
    print("NATURE DATASET TRAINING")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Mixed Precision: {config.USE_MIXED_PRECISION}")
    print(f"Data Split: Train={config.TRAIN_SPLIT}, Val={config.VAL_SPLIT}, Test={config.TEST_SPLIT}")
    print("="*60)
    
    # Load full dataset
    print("\nLoading Nature dataset...")
    full_dataset = NatureDataset(config.NATURE_DIR, transform=None)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(config.TRAIN_SPLIT * total_size)
    val_size = int(config.VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nDataset split:")
    print(f"  Train: {train_size} images ({config.TRAIN_SPLIT*100:.0f}%)")
    print(f"  Val:   {val_size} images ({config.VAL_SPLIT*100:.0f}%)")
    print(f"  Test:  {test_size} images ({config.TEST_SPLIT*100:.0f}%)")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = get_transforms(train=True)
    val_dataset.dataset.transform = get_transforms(train=False)
    test_dataset.dataset.transform = get_transforms(train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.PIN_MEMORY and getattr(config, 'CUDA_AVAILABLE', False)),
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.PIN_MEMORY and getattr(config, 'CUDA_AVAILABLE', False)),
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.PIN_MEMORY and getattr(config, 'CUDA_AVAILABLE', False)),
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False
    )
    
    # Create model
    model = create_model()
    print(f"Model created and moved to {config.DEVICE}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Learning rate scheduler: use ReduceLROnPlateau to reduce LR when val loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True,
        min_lr=1e-6
    )

    # Mixed precision scaler: only enable AMP when CUDA is available
    use_amp = config.USE_MIXED_PRECISION and getattr(config, 'CUDA_AVAILABLE', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\nStarting training...\n")
    
    # Training loop
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, epoch, "Val")
        
        # Step scheduler
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            model_path = config.MODEL_DIR / f"{config.MODEL_SAVE_NAME}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': vars(config)
            }, model_path)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  Early stopping patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
        
        # Save checkpoint every epoch
        checkpoint_path = config.MODEL_DIR / f"{config.MODEL_SAVE_NAME}_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        print("-" * 60)
    
    # Test on test set
    print("\n" + "="*60)
    print("TESTING ON TEST SET")
    print("="*60)
    
    # Load best model
    best_model_path = config.MODEL_DIR / f"{config.MODEL_SAVE_NAME}_best.pth"
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    test_loss, test_acc = validate(model, test_loader, criterion, epoch, "Test")
    
    print(f"\nFinal Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    
    # Save final history
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    history['best_val_acc'] = best_val_acc
    
    history_path = config.RESULTS_DIR / f"{config.MODEL_SAVE_NAME}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\nTraining complete!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Test accuracy: {test_acc:.2f}%")
    print(f"  Model saved as: {config.MODEL_SAVE_NAME}_best.pth")
    print(f"  History saved to: {history_path}")
    print("="*60)

if __name__ == "__main__":
    main()
