"""
Training script for Buildings Deepfake Detection
Uses EfficientNet-B3 with heavy augmentation for small dataset
"""

import warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import time
import json

import config_buildings as config
from dataset_buildings import create_buildings_dataloaders
from utils import (
    set_seed, calculate_metrics, print_metrics,
    save_checkpoint, EarlyStopping
)


class BuildingsClassifier(nn.Module):
    """EfficientNet-B3 based classifier for buildings"""
    
    def __init__(self, num_classes=1, pretrained=True, dropout_rate=0.5):
        super(BuildingsClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B3
        self.backbone = models.efficientnet_b3(weights='DEFAULT' if pretrained else None)
        
        # Get number of features
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        print(f" BuildingsClassifier created")
        print(f"  Backbone: EfficientNet-B3")
        print(f"  Pretrained: {pretrained}")
        print(f"  Output classes: {num_classes}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if config.GRADIENT_CLIPPING > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIPPING)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        with torch.no_grad():
            scores = torch.sigmoid(outputs).cpu().numpy()
            preds = (scores > 0.5).astype(int)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.flatten())
            all_scores.extend(scores.flatten())
        
        # Update progress bar
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Calculate metrics
    avg_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores)
    )
    
    return avg_loss, metrics


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            
            running_loss += loss.item()
            
            scores = torch.sigmoid(outputs).cpu().numpy()
            preds = (scores > 0.5).astype(int)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.flatten())
            all_scores.extend(scores.flatten())
    
    avg_loss = running_loss / len(val_loader)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores)
    )
    
    return avg_loss, metrics


def train_buildings():
    """Main training function"""
    print("\n" + "="*80)
    print("BUILDINGS DEEPFAKE DETECTION - TRAINING")
    print("="*80 + "\n")
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_buildings_dataloaders()
    
    # Create model
    print(f"\n{'='*80}")
    print("CREATING MODEL")
    print(f"{'='*80}\n")
    
    model = BuildingsClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)
    
    total_params, trainable_params = model.get_num_params()
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Loss function with class weights
    pos_weight = torch.tensor([config.CLASS_WEIGHTS[1] / config.CLASS_WEIGHTS[0]]).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"\nLoss: Weighted BCE (class weights: {config.CLASS_WEIGHTS})")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    print(f"Optimizer: AdamW (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.SCHEDULER_PARAMS['T_max'],
        eta_min=config.SCHEDULER_PARAMS['eta_min']
    )
    print(f"Scheduler: CosineAnnealing")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        
        # Training
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, config.DEVICE
        )
        
        # Validation
        val_loss, val_metrics = validate(model, val_loader, criterion, config.DEVICE)
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['learning_rates'].append(current_lr)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} Summary (Time: {epoch_time:.1f}s)")
        print(f"{'='*80}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['accuracy']*100:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"\nValidation Metrics:")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall:    {val_metrics['recall']:.4f}")
        print(f"  F1 Score:  {val_metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {val_metrics.get('roc_auc', 0.0):.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_val_acc,
                'val_loss': val_loss,
                'config': {
                    'model_name': config.MODEL_NAME,
                    'img_size': config.IMG_SIZE,
                    'num_classes': config.NUM_CLASSES
                }
            }
            
            save_path = config.MODEL_DIR / 'best_model_buildings.pth'
            torch.save(checkpoint, save_path)
            print(f"\n✓ Best model saved! (Val Acc: {best_val_acc*100:.2f}%)")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = config.MODEL_DIR / f'checkpoint_buildings_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_metrics['accuracy'],
                'val_loss': val_loss
            }, checkpoint_path)
        
        print(f"{'='*80}\n")
        
        # Early stopping
        early_stopping(val_loss, model, epoch + 1)
        if early_stopping.early_stop:
            print(f"\n{'='*80}")
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"{'='*80}\n")
            break
    
    # Training complete
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to: {config.MODEL_DIR / 'best_model_buildings.pth'}")
    
    # Save training history
    history_path = config.RESULTS_DIR / 'training_history_buildings.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    print(f"{'='*80}\n")
    
    return model, history


if __name__ == "__main__":
    try:
        model, history = train_buildings()
        print("\n Training completed successfully!")
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
    except Exception as e:
        print(f"\n Error during training: {str(e)}")
        raise e
