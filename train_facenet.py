"""
Training script for FaceNet-based Face Deepfake Detection
Optimized for Real vs Fake face classification
"""

import warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
import json

import config_facenet as config
from model_facenet import create_facenet_model
from dataset_facenet import create_facenet_dataloaders
from utils import (
    set_seed, calculate_metrics, print_metrics,
    save_checkpoint, EarlyStopping
)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets.float().unsqueeze(1), reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class HardExampleMiningLoss(nn.Module):
    """BCE Loss with hard example mining - focuses on difficult samples"""
    def __init__(self, ratio=0.3, class_weights=None):
        super().__init__()
        self.ratio = ratio
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        # Calculate per-sample loss
        if self.class_weights is not None:
            # Apply class weights
            weights = torch.tensor([self.class_weights[int(t)] for t in targets], 
                                  device=inputs.device).unsqueeze(1)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                inputs, targets.float().unsqueeze(1), reduction='none'
            ) * weights
        else:
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                inputs, targets.float().unsqueeze(1), reduction='none'
            )
        
        # Select hardest examples
        num_hard = max(1, int(self.ratio * len(bce_loss)))
        hard_losses, _ = torch.topk(bce_loss.squeeze(), num_hard)
        
        return hard_losses.mean()


def get_loss_function():
    """Get loss function based on config"""
    if config.LOSS_TYPE == 'focal':
        print(f"  Using Focal Loss (alpha={config.FOCAL_LOSS_ALPHA}, gamma={config.FOCAL_LOSS_GAMMA})")
        return FocalLoss(
            alpha=config.FOCAL_LOSS_ALPHA,
            gamma=config.FOCAL_LOSS_GAMMA
        )
    elif config.LOSS_TYPE == 'weighted_bce':
        print(f"  Using Weighted BCE with class weights: {config.CLASS_WEIGHTS}")
        if config.USE_HARD_EXAMPLE_MINING:
            print(f"  + Hard Example Mining (ratio={config.HARD_EXAMPLE_RATIO})")
            return HardExampleMiningLoss(
                ratio=config.HARD_EXAMPLE_RATIO,
                class_weights=config.CLASS_WEIGHTS
            )
        else:
            # Standard weighted BCE
            pos_weight = torch.tensor([config.CLASS_WEIGHTS[1] / config.CLASS_WEIGHTS[0]]).to(config.DEVICE)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        print(f"  Using standard BCE Loss")
        return nn.BCEWithLogitsLoss()


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
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Apply label smoothing manually if needed
        if config.LABEL_SMOOTHING > 0:
            labels_smooth = labels.float() * (1 - config.LABEL_SMOOTHING) + 0.5 * config.LABEL_SMOOTHING
            loss = nn.functional.binary_cross_entropy_with_logits(
                outputs.squeeze(), labels_smooth
            )
        else:
            loss = criterion(outputs.squeeze(), labels.float())
        
        # Backward pass
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
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss - handle different loss function types
            if isinstance(criterion, HardExampleMiningLoss):
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs.squeeze(), labels.float())
            
            running_loss += loss.item()
            
            # Get predictions
            scores = torch.sigmoid(outputs).cpu().numpy()
            preds = (scores > 0.5).astype(int)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.flatten())
            all_scores.extend(scores.flatten())
    
    # Calculate metrics
    avg_loss = running_loss / len(val_loader)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores)
    )
    
    return avg_loss, metrics


def train_facenet():
    """Main training function"""
    print("\n" + "="*80)
    print("FACENET FACE DEEPFAKE DETECTION - TRAINING")
    print("="*80 + "\n")
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_facenet_dataloaders()
    
    # Create model
    print(f"\n{'='*80}")
    print("CREATING MODEL")
    print(f"{'='*80}\n")
    
    use_attention = config.MODEL_TYPE == 'facenet_attention'
    model = create_facenet_model(use_attention=use_attention)
    
    # Loss function
    criterion = get_loss_function()
    
    # Optimizer - Using AdamW with better weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    print(f"Optimizer: AdamW (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
    
    # Learning rate scheduler with warmup
    if config.SCHEDULER_TYPE == 'cosine_warmup':
        # Cosine annealing with warmup
        warmup_epochs = config.SCHEDULER_PARAMS.get('warmup_epochs', 3)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing after warmup
                progress = (epoch - warmup_epochs) / (config.NUM_EPOCHS - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"Scheduler: Cosine Annealing with {warmup_epochs}-epoch warmup")
        
    elif config.SCHEDULER_TYPE == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.SCHEDULER_PARAMS['T_max'],
            eta_min=config.SCHEDULER_PARAMS['eta_min']
        )
    elif config.SCHEDULER_TYPE == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.LEARNING_RATE,
            epochs=config.NUM_EPOCHS,
            steps_per_epoch=len(train_loader)
        )
    elif config.SCHEDULER_TYPE == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
    else:
        scheduler = None
    
    if scheduler is not None:
        print(f"Scheduler: {config.SCHEDULER_TYPE}")
    else:
        print(f"No learning rate scheduler")
    
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
        if scheduler is not None:
            if config.SCHEDULER_TYPE == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
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
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
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
                    'num_classes': config.NUM_CLASSES,
                    'pretrained': config.FACENET_PRETRAINED
                }
            }
            
            save_path = config.MODEL_DIR / 'best_model_facenet.pth'
            torch.save(checkpoint, save_path)
            print(f"\n✓ Best model saved! (Val Acc: {best_val_acc:.2f}%)")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = config.MODEL_DIR / f'checkpoint_facenet_epoch_{epoch+1}.pth'
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_metrics['accuracy'],
                'val_loss': val_loss
            }
            torch.save(checkpoint, checkpoint_path)
        
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
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {config.MODEL_DIR / 'best_model_facenet.pth'}")
    
    # Save training history
    history_path = config.RESULTS_DIR / 'training_history_facenet.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    print(f"{'='*80}\n")
    
    return model, history


if __name__ == "__main__":
    try:
        model, history = train_facenet()
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")
    except Exception as e:
        print(f"\n\n Error during training: {str(e)}")
        raise e
