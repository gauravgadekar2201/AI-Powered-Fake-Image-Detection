"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Generates visual explanations for model predictions by highlighting important regions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for generating visual explanations
    Works with any CNN architecture
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Layer to compute gradients for (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heat map
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Class index to generate CAM for (None = predicted class)
            
        Returns:
            cam: Heat map as numpy array (H, W) with values 0-1
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def create_heatmap_overlay(self, original_image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Create heat map overlay on original image
        
        Args:
            original_image: PIL Image or numpy array
            cam: Heat map from generate_cam (H, W)
            alpha: Transparency (0=only heatmap, 1=only image)
            colormap: OpenCV colormap
            
        Returns:
            overlay: PIL Image with heat map overlay
        """
        # Convert PIL to numpy if needed
        if isinstance(original_image, Image.Image):
            img = np.array(original_image)
        else:
            img = original_image.copy()
        
        # Resize CAM to match image size
        h, w = img.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = (alpha * img + (1 - alpha) * heatmap).astype(np.uint8)
        
        return Image.fromarray(overlay)
    
    def get_explanation_text(self, cam, threshold=0.7):
        """
        Generate textual explanation from heat map
        
        Args:
            cam: Heat map from generate_cam
            threshold: Threshold for "suspicious" regions (0-1)
            
        Returns:
            explanation: Text describing suspicious regions
        """
        # Find suspicious regions
        suspicious_mask = cam > threshold
        
        if not suspicious_mask.any():
            return "Model confidence is distributed across the entire image."
        
        # Calculate statistics
        suspicious_percentage = (suspicious_mask.sum() / cam.size) * 100
        max_activation = cam.max()
        
        # Generate explanation
        explanations = []
        
        if suspicious_percentage > 50:
            explanations.append(f"⚠️ Large suspicious region detected ({suspicious_percentage:.1f}% of image)")
        elif suspicious_percentage > 20:
            explanations.append(f"⚠️ Multiple suspicious regions found ({suspicious_percentage:.1f}% of image)")
        else:
            explanations.append(f"⚠️ Small focused suspicious region ({suspicious_percentage:.1f}% of image)")
        
        explanations.append(f"🎯 Maximum activation: {max_activation:.2f}")
        
        # Describe location (rough quadrant analysis)
        h, w = cam.shape
        top_half = cam[:h//2, :].sum()
        bottom_half = cam[h//2:, :].sum()
        left_half = cam[:, :w//2].sum()
        right_half = cam[:, w//2:].sum()
        
        vertical = "upper" if top_half > bottom_half else "lower"
        horizontal = "left" if left_half > right_half else "right"
        
        explanations.append(f"📍 Strongest in {vertical}-{horizontal} region")
        
        return " | ".join(explanations)


def get_target_layer(model, model_type):
    """
    Get the appropriate target layer for Grad-CAM based on model type
    
    Args:
        model: PyTorch model
        model_type: 'facenet', 'buildings', or 'nature'
        
    Returns:
        target_layer: Layer to use for Grad-CAM
    """
    if model_type == 'facenet':
        # InceptionResnetV1 - last conv layer in mixed_8
        return model.model.mixed_8_branch_1[-1]
    
    elif model_type == 'buildings':
        # EfficientNet-B4 - last conv layer before pooling
        if hasattr(model.efficientnet, 'features'):
            return model.efficientnet.features[-1]
        else:
            return model.efficientnet.conv_head
    
    elif model_type == 'nature':
        # EfficientNet-B3 - last conv layer before pooling
        if hasattr(model, 'features'):
            return model.features[-1]
        elif hasattr(model, 'efficientnet'):
            if hasattr(model.efficientnet, 'features'):
                return model.efficientnet.features[-1]
            else:
                return model.efficientnet.conv_head
        else:
            return model.features[-1]
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_gradcam_explanation(model, input_tensor, original_image, model_type):
    """
    Complete pipeline: Generate Grad-CAM and explanation
    
    Args:
        model: PyTorch model
        input_tensor: Preprocessed input tensor
        original_image: Original PIL Image
        model_type: 'facenet', 'buildings', or 'nature'
        
    Returns:
        dict with 'heatmap_image' (PIL Image) and 'explanation' (str)
    """
    try:
        # Get target layer
        target_layer = get_target_layer(model, model_type)
        
        # Create Grad-CAM
        gradcam = GradCAM(model, target_layer)
        
        # Generate CAM
        cam = gradcam.generate_cam(input_tensor)
        
        # Create overlay
        heatmap_overlay = gradcam.create_heatmap_overlay(original_image, cam, alpha=0.6)
        
        # Generate explanation
        explanation = gradcam.get_explanation_text(cam, threshold=0.7)
        
        return {
            'heatmap_image': heatmap_overlay,
            'explanation': explanation,
            'success': True
        }
    
    except Exception as e:
        return {
            'heatmap_image': original_image,
            'explanation': f"Could not generate explanation: {str(e)}",
            'success': False
        }
