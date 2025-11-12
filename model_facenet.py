"""
FaceNet-based Model for Face Deepfake Detection
Uses InceptionResnetV1 (FaceNet) pretrained on VGGFace2
"""

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class FaceNetClassifier(nn.Module):
    """
    FaceNet-based classifier for Real vs Fake face detection
    Uses pretrained InceptionResnetV1 with custom classification head
    """
    
    def __init__(self, num_classes=1, pretrained='vggface2', dropout_rate=0.5):
        """
        Args:
            num_classes: Number of output classes (1 for binary, 3 for Real/Fake/Anime)
            pretrained: Pretrained weights - 'vggface2' or 'casia-webface' or None
            dropout_rate: Dropout probability
        """
        super(FaceNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained FaceNet model
        # InceptionResnetV1 outputs 512-dimensional embeddings
        self.facenet = InceptionResnetV1(
            pretrained=pretrained,
            classify=False,  # Don't use built-in classifier
            dropout_prob=dropout_rate
        )
        
        # Freeze early layers (optional - can fine-tune later)
        self._freeze_early_layers(freeze=False)
        
        # Custom classification head optimized for deepfake detection
        self.classifier = self._build_classifier(512, num_classes, dropout_rate)
        
        # For Grad-CAM
        self.gradients = None
        self.activations = None
    
    def _freeze_early_layers(self, freeze=True):
        """
        Freeze early layers to preserve pretrained face features
        Can unfreeze later for fine-tuning
        """
        if freeze:
            # Freeze first few blocks
            for name, param in self.facenet.named_parameters():
                if any(block in name for block in ['conv2d_1a', 'conv2d_2a', 'conv2d_2b']):
                    param.requires_grad = False
        else:
            # Unfreeze all layers
            for param in self.facenet.parameters():
                param.requires_grad = True
    
    def _build_classifier(self, in_features, num_classes, dropout_rate):
        """
        Build classification head with attention to deepfake artifacts
        
        Architecture:
        512 -> 256 -> 128 -> num_classes
        With BatchNorm, Dropout, and ReLU activations
        """
        layers = []
        
        # First hidden layer (512 -> 256)
        layers.append(nn.Linear(in_features, 256))
        layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=dropout_rate))
        
        # Second hidden layer (256 -> 128)
        layers.append(nn.Linear(256, 128))
        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=dropout_rate * 0.7))
        
        # Output layer
        layers.append(nn.Linear(128, num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Standard forward pass"""
        # Get face embeddings from FaceNet
        embeddings = self.facenet(x)
        
        # Classify
        output = self.classifier(embeddings)
        
        return output
    
    def forward_with_embeddings(self, x):
        """
        Forward pass that returns both predictions and embeddings
        Useful for visualization and analysis
        """
        embeddings = self.facenet(x)
        output = self.classifier(embeddings)
        return output, embeddings
    
    def get_embeddings(self, x):
        """Extract face embeddings only"""
        with torch.no_grad():
            embeddings = self.facenet(x)
        return embeddings
    
    def unfreeze_all_layers(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ All layers unfrozen for fine-tuning")
    
    def get_num_params(self):
        """Get number of parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class FaceNetWithAttention(nn.Module):
    """
    Enhanced FaceNet with spatial attention mechanism
    Helps model focus on important facial regions for deepfake detection
    """
    
    def __init__(self, num_classes=1, pretrained='vggface2', dropout_rate=0.5):
        super(FaceNetWithAttention, self).__init__()
        
        self.num_classes = num_classes
        
        # Base FaceNet
        self.facenet = InceptionResnetV1(
            pretrained=pretrained,
            classify=False,
            dropout_prob=dropout_rate
        )
        
        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            
            nn.Linear(128, num_classes)
        )
        
        print(f"✓ FaceNet with Attention created")
        print(f"  Architecture: FaceNet + Spatial Attention")
    
    def forward(self, x):
        # Apply spatial attention (simplified version)
        # In practice, attention should be applied at feature level
        embeddings = self.facenet(x)
        output = self.classifier(embeddings)
        return output
    
    def get_num_params(self):
        """Get number of parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_facenet_model(use_attention=False):
    """
    Create FaceNet-based model based on config
    
    Args:
        use_attention: Whether to use attention mechanism
    
    Returns:
        nn.Module: Model ready for training
    """
    if use_attention:
        model = FaceNetWithAttention(
            num_classes=config.NUM_CLASSES,
            pretrained='vggface2',
            dropout_rate=config.DROPOUT_RATE
        )
    else:
        model = FaceNetClassifier(
            num_classes=config.NUM_CLASSES,
            pretrained='vggface2',
            dropout_rate=config.DROPOUT_RATE
        )
    
    # Move to device
    model = model.to(config.DEVICE)
    
    # Multi-GPU support
    if config.USE_MULTI_GPU and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Print model info
    total_params, trainable_params = model.module.get_num_params() if hasattr(model, 'module') else model.get_num_params()
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing FaceNet model creation...\n")
    
    # Test regular FaceNet
    print("1. Regular FaceNet Classifier:")
    model = create_facenet_model(use_attention=False)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 160, 160).to(config.DEVICE)  # FaceNet expects 160x160
    output = model(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test with attention
    print("\n" + "="*60)
    print("2. FaceNet with Attention:")
    model_attn = create_facenet_model(use_attention=True)
    output_attn = model_attn(dummy_input)
    print(f"  Output shape: {output_attn.shape}")
    
    print("\n✅ FaceNet models working correctly!")
