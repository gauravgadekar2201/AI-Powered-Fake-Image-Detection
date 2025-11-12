"""
Nature Detection Model
Uses EfficientNet-B3 for classifying real vs AI-generated nature images
"""

import torch
import torch.nn as nn
import torchvision.models as models


class NatureClassifier(nn.Module):
    """
    EfficientNet-B3 based classifier for nature deepfake detection
    """
    
    def __init__(self, num_classes=1, pretrained=True, dropout_rate=0.5):
        """
        Args:
            num_classes: Number of output classes (1 for binary classification)
            pretrained: Use pretrained ImageNet weights
            dropout_rate: Dropout probability for regularization
        """
        super(NatureClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained EfficientNet-B3  
        efficientnet = models.efficientnet_b3(weights='DEFAULT' if pretrained else None)
        
        # Get number of features from the backbone
        in_features = efficientnet.classifier[1].in_features
        
        # Extract features separately to match training structure
        self.features = efficientnet.features
        
        # Use EfficientNet's built-in avgpool
        self.avgpool = efficientnet.avgpool
        
        # Replace classifier with custom head
        # Simpler architecture for nature: in_features -> num_classes
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_num_params(self):
        """Get number of parameters in the model"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_nature_model(num_classes=1, pretrained=True, dropout_rate=0.5, device='cpu'):
    """
    Create and return a NatureClassifier model
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout_rate: Dropout probability
        device: Device to load model on
    
    Returns:
        NatureClassifier model
    """
    model = NatureClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    
    model = model.to(device)
    
    # Print model info
    total_params, trainable_params = model.get_num_params()
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing NatureClassifier...\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create model
    model = create_nature_model(
        num_classes=1,
        pretrained=True,
        dropout_rate=0.5,
        device=device
    )
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output values: {output.squeeze().detach().cpu().numpy()}")
    
    print("\n✅ NatureClassifier working correctly!")
