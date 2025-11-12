"""
Buildings Detection Model
Uses EfficientNet-B4 for classifying real vs AI-generated building images
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BuildingsClassifier(nn.Module):
    """
    EfficientNet-B4 based classifier for buildings deepfake detection
    """
    
    def __init__(self, num_classes=1, pretrained=True, dropout_rate=0.7):
        """
        Args:
            num_classes: Number of output classes (1 for binary classification)
            pretrained: Use pretrained ImageNet weights
            dropout_rate: Dropout probability for regularization
        """
        super(BuildingsClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained EfficientNet-B4
        self.backbone = models.efficientnet_b4(weights='DEFAULT' if pretrained else None)
        
        # Get number of features from the backbone
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        # Architecture: in_features -> 512 -> 256 -> num_classes
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
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.backbone(x)
    
    def get_num_params(self):
        """Get number of parameters in the model"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_buildings_model(num_classes=1, pretrained=True, dropout_rate=0.7, device='cpu'):
    """
    Create and return a BuildingsClassifier model
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout_rate: Dropout probability
        device: Device to load model on
    
    Returns:
        BuildingsClassifier model
    """
    model = BuildingsClassifier(
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
    print("Testing BuildingsClassifier...\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create model
    model = create_buildings_model(
        num_classes=1,
        pretrained=True,
        dropout_rate=0.7,
        device=device
    )
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output values: {output.squeeze().detach().cpu().numpy()}")
    
    print("\n BuildingsClassifier working correctly!")
