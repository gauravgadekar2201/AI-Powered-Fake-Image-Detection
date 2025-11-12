"""
Evaluation script for AI Image Detection Model
Evaluates the trained model on the test set
"""

import torch
import numpy as np
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import seaborn as sns

import config
from dataset import create_dataloaders
from model import create_model
from utils import load_checkpoint, calculate_metrics, print_metrics


def evaluate_model(model, test_loader):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
    
    Returns:
        dict: Evaluation metrics and predictions
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_scores = []
    
    print("\nEvaluating model on test set...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(config.DEVICE)
            labels = labels.numpy()
            
            # Forward pass
            if config.USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            # Get predictions
            scores = torch.sigmoid(outputs).cpu().numpy()
            preds = (scores > 0.5).astype(int)
            
            all_labels.extend(labels)
            all_preds.extend(preds.flatten())
            all_scores.extend(scores.flatten())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {(batch_idx + 1) * config.BATCH_SIZE} / {len(test_loader.dataset)} images")
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_scores)
    
    return {
        'metrics': metrics,
        'labels': all_labels,
        'predictions': all_preds,
        'scores': all_scores
    }


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_score_distribution(labels, scores, save_path=None):
    """Plot distribution of prediction scores"""
    plt.figure(figsize=(10, 6))
    
    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]
    
    plt.hist(real_scores, bins=50, alpha=0.5, label='Real Images', color='blue')
    plt.hist(fake_scores, bins=50, alpha=0.5, label='Fake Images', color='red')
    
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Prediction Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Score distribution saved to: {save_path}")
    
    plt.show()


def main():
    """Main evaluation function"""
    print("\n" + "="*60)
    print("AI IMAGE DETECTION - EVALUATION")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_loader = create_dataloaders()
    
    # Create model
    print("\nCreating model...")
    model = create_model()
    
    # Load best checkpoint
    checkpoint_path = config.MODEL_DIR / 'best_model.pth'
    if checkpoint_path.exists():
        print(f"\nLoading best model from: {checkpoint_path}")
        model, _, epoch, val_acc = load_checkpoint(model, None, checkpoint_path)
    else:
        print(f"\nError: No checkpoint found at {checkpoint_path}")
        return
    
    # Evaluate
    results = evaluate_model(model, test_loader)
    
    # Print metrics
    print_metrics(results['metrics'], "Test Set Evaluation")
    
    # Plot confusion matrix
    cm = np.array(results['metrics']['confusion_matrix'])
    plot_confusion_matrix(
        cm,
        save_path=config.RESULTS_DIR / 'confusion_matrix.png'
    )
    
    # Plot score distribution
    plot_score_distribution(
        results['labels'],
        results['scores'],
        save_path=config.RESULTS_DIR / 'score_distribution.png'
    )
    
    # Calculate per-class accuracy
    tn, fp, fn, tp = cm.ravel()
    real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nPer-Class Accuracy:")
    print(f"  Real Images: {real_acc*100:.2f}%")
    print(f"  Fake Images: {fake_acc*100:.2f}%")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
