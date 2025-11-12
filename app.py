"""
Unified Flask Backend for Multi-Model AI Image Detection
Supports: FaceNet (faces), Buildings, and Nature detection models
Features: Auto-detection, Manual selection, Ensemble mode
"""

import os
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import cv2
import base64
from io import BytesIO

# Import face detection
from facenet_pytorch import MTCNN

# Import Grad-CAM for explanations
from utils_gradcam import generate_gradcam_explanation

# Import model configurations
import config_facenet
import config_buildings

# Import model architectures
from model_facenet import FaceNetClassifier
from model_buildings import BuildingsClassifier
from model_nature import NatureClassifier

# ===============================
# FLASK APP CONFIGURATION
# ===============================
app = Flask(__name__, 
            static_folder='gui',
            static_url_path='')

UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'jfif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ===============================
# MODEL MANAGER CLASS
# ===============================
class ModelManager:
    """
    Manages all 3 detection models (FaceNet, Buildings, Nature)
    Handles: Loading, Auto-detection, Prediction, Ensemble
    """
    
    def __init__(self, device):
        self.device = device
        
        # Initialize models
        self.facenet_model = None
        self.buildings_model = None
        self.nature_model = None
        self.mtcnn = None
        
        # Load all models silently
        import sys
        import io
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            self.load_all_models()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def load_facenet(self):
        """Load FaceNet model for face detection"""
        print("Loading FaceNet Model...")
        try:
            model_path = config_facenet.MODEL_DIR / 'best_model_facenet.pth'
            
            if not model_path.exists():
                print(f"  ⚠️  Model not found: {model_path}")
                return None
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            model_config = checkpoint.get('config', {})
            
            # Create model
            model = FaceNetClassifier(
                num_classes=model_config.get('num_classes', 1),
                pretrained=model_config.get('pretrained', 'vggface2'),
                dropout_rate=config_facenet.DROPOUT_RATE
            ).to(self.device)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Initialize MTCNN for face detection
            self.mtcnn = MTCNN(
                image_size=config_facenet.IMG_SIZE,
                margin=config_facenet.FACE_DETECTION_MARGIN,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.9],
                keep_all=False,
                device=self.device
            )
            
            print(f"  ✓ FaceNet loaded (Accuracy: {checkpoint.get('accuracy', 'N/A')})")
            print(f"  ✓ MTCNN face detector initialized")
            return model
            
        except Exception as e:
            print(f"  ❌ Error loading FaceNet: {e}")
            return None
    
    def load_buildings(self):
        """Load buildings detection model"""
        print("Loading Buildings Model...")
        try:
            import warnings
            import sys
            import io
            from contextlib import redirect_stderr
            from model_buildings import BuildingsClassifier
            model = BuildingsClassifier(pretrained=False).to(self.device)
            
            model_path = os.path.join('output', 'models', 'best_model_buildings.pth')
            if not os.path.exists(model_path):
                print(f"  ❌ Buildings model not found at: {model_path}")
                return None
            
            # Suppress size mismatch warnings by redirecting stderr
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # Redirect stderr to suppress PyTorch warnings
                f = io.StringIO()
                with redirect_stderr(f):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            model.eval()
            
            accuracy = checkpoint.get('accuracy', 0)
            if isinstance(accuracy, (int, float)):
                print(f"  ✓ Buildings loaded (Accuracy: {accuracy:.1f}%)")
            else:
                print(f"  ✓ Buildings loaded")
            return model
            
        except Exception as e:
            print(f"  ❌ Error loading Buildings: {e}")
            return None
    
    def load_nature(self):
        """Load Nature detection model"""
        print("Loading Nature Model...")
        try:
            import warnings
            import sys
            import io
            from contextlib import redirect_stderr
            model_path = config_buildings.MODEL_DIR / 'nature_best.pth'
            
            if not model_path.exists():
                print(f"  ⚠️  Model not found: {model_path}")
                return None
            
            # Suppress size mismatch warnings by redirecting stderr
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # Redirect stderr to suppress PyTorch warnings
                f = io.StringIO()
                with redirect_stderr(f):
                    # Load checkpoint
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    
                    # Create model
                    model = NatureClassifier(
                        num_classes=1,
                        pretrained=False,  # Already trained
                        dropout_rate=0.5
                    ).to(self.device)
                    
                    # Load weights with strict=False to ignore mismatches
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.eval()
            
            accuracy = checkpoint.get('val_acc', 0)
            if isinstance(accuracy, (int, float)):
                print(f"  ✓ Nature loaded (Accuracy: {accuracy:.1f}%)")
            else:
                print(f"  ✓ Nature loaded")
            return model
            
        except Exception as e:
            print(f"  ❌ Error loading Nature: {e}")
            return None
    
    def load_all_models(self):
        """Load all 3 models"""
        self.facenet_model = self.load_facenet()
        self.buildings_model = self.load_buildings()
        self.nature_model = self.load_nature()
    
    def detect_face(self, image):
        """
        Detect face in image using MTCNN
        Returns: cropped face PIL Image or None
        """
        if self.mtcnn is None:
            return None
        
        try:
            face = self.mtcnn(image)
            
            if face is not None:
                # Convert tensor to PIL Image
                face = face.permute(1, 2, 0).cpu().numpy()
                face = ((face * 128) + 128).clip(0, 255).astype('uint8')
                face = Image.fromarray(face)
                return face
            return None
        except Exception as e:
            print(f"Face detection error: {e}")
            return None
    
    def detect_building(self, image):
        """
        Detect if image contains building using heuristics
        Returns: confidence score (0.0 to 1.0)
        """
        try:
            # Convert PIL to numpy array
            img_array = np.array(image.convert('RGB'))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect lines (buildings have many straight lines)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                    minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                num_lines = len(lines)
                # More lines = more likely a building
                confidence = min(num_lines / 100.0, 1.0)
                return confidence
            
            return 0.0
            
        except Exception as e:
            print(f"Building detection error: {e}")
            return 0.0
    
    def detect_image_type(self, image):
        """
        Auto-detect image type (face, building, or nature)
        Returns: ('face'|'building'|'nature', confidence)
        """
        # Priority 1: Face detection
        face = self.detect_face(image)
        if face is not None:
            return 'face', 0.95
        
        # Priority 2: Building detection
        building_confidence = self.detect_building(image)
        if building_confidence > 0.3:
            return 'building', building_confidence
        
        # Priority 3: Nature (default)
        return 'nature', 0.60
    
    def preprocess_image(self, image, model_type):
        """
        Preprocess image for specific model
        Returns: tensor ready for model
        """
        if model_type == 'face':
            # FaceNet uses 160x160 with specific normalization
            transform = transforms.Compose([
                transforms.Resize((config_facenet.IMG_SIZE, config_facenet.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            # Buildings and Nature use 224x224 with ImageNet normalization
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def predict_single(self, image, model_type='auto', generate_explanation=False):
        """
        Make prediction using single model
        
        Args:
            image: PIL Image
            model_type: 'auto', 'face', 'building', or 'nature'
            generate_explanation: Whether to generate Grad-CAM explanation
        
        Returns:
            dict with prediction results and optional explanation
        """
        # Auto-detect if needed
        auto_detected = False
        if model_type == 'auto':
            model_type, detection_confidence = self.detect_image_type(image)
            auto_detected = True
        
        # Store original image for Grad-CAM
        original_image = image.copy()
        
        # Select model
        if model_type == 'face' and self.facenet_model is not None:
            model = self.facenet_model
            model_name = "FaceNet (VGGFace2)"
            
            # Try to detect and crop face
            face = self.detect_face(image)
            face_detected = face is not None
            if face_detected:
                image = face
            else:
                image = image.resize((config_facenet.IMG_SIZE, config_facenet.IMG_SIZE))
        
        elif model_type == 'building' and self.buildings_model is not None:
            model = self.buildings_model
            model_name = "Buildings Detector (EfficientNet-B4)"
            face_detected = False
        
        elif model_type == 'nature' and self.nature_model is not None:
            model = self.nature_model
            model_name = "Nature Detector (EfficientNet-B3)"
            face_detected = False
        
        else:
            return {
                'error': f'Model "{model_type}" not available',
                'success': False
            }
        
        # Preprocess
        image_tensor = self.preprocess_image(image, model_type)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probability = torch.sigmoid(output).item()
        
        # Interpret results
        if probability > 0.5:
            prediction = "FAKE"
            confidence_fake = probability * 100
            confidence_str = f"{confidence_fake:.1f}% Fake"
        else:
            prediction = "REAL"
            confidence_real = (1 - probability) * 100
            confidence_str = f"{confidence_real:.1f}% Real"
        
        result = {
            'success': True,
            'prediction': prediction,
            'confidence': confidence_str,
            'probability': round(probability, 4),
            'model_used': model_type,
            'model_name': model_name,
            'auto_detected': auto_detected,
            'face_detected': face_detected if model_type == 'face' else None
        }
        
        # Generate Grad-CAM explanation if requested
        if generate_explanation:
            gradcam_result = generate_gradcam_explanation(
                model, image_tensor, image, model_type
            )
            
            if gradcam_result['success']:
                # Convert PIL Image to base64
                buffered = BytesIO()
                gradcam_result['heatmap_image'].save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                result['explanation'] = {
                    'heatmap': f"data:image/png;base64,{img_str}",
                    'text': gradcam_result['explanation']
                }
            else:
                result['explanation'] = {
                    'heatmap': None,
                    'text': gradcam_result['explanation']
                }
        
        return result
    
    def predict_ensemble(self, image, generate_explanation=False):
        """
        Run all 3 models and return ensemble result
        
        Args:
            image: PIL Image
            generate_explanation: Whether to generate Grad-CAM explanations
        
        Returns:
            dict with all model results, consensus, and optional explanations
        """
        results = {}
        
        # Run each model
        for model_type in ['face', 'building', 'nature']:
            result = self.predict_single(image, model_type, generate_explanation)
            if result['success']:
                results[model_type] = {
                    'prediction': result['prediction'],
                    'confidence': result['probability'],
                    'confidence_str': result['confidence'],
                    'model_name': result['model_name']
                }
                
                # Add explanation if generated
                if generate_explanation and 'explanation' in result:
                    results[model_type]['explanation'] = result['explanation']
        
        # Calculate consensus
        if len(results) > 0:
            # Weighted average of probabilities
            total_prob = sum(r['confidence'] for r in results.values())
            avg_prob = total_prob / len(results)
            
            # Count votes
            fake_votes = sum(1 for r in results.values() if r['prediction'] == 'FAKE')
            real_votes = sum(1 for r in results.values() if r['prediction'] == 'REAL')
            
            # Consensus
            if avg_prob > 0.5:
                consensus_prediction = "FAKE"
                consensus_confidence = avg_prob * 100
                consensus_str = f"{consensus_confidence:.1f}% Fake"
            else:
                consensus_prediction = "REAL"
                consensus_confidence = (1 - avg_prob) * 100
                consensus_str = f"{consensus_confidence:.1f}% Real"
            
            consensus = {
                'prediction': consensus_prediction,
                'confidence': round(avg_prob, 4),
                'confidence_str': consensus_str,
                'agreement': f"{max(fake_votes, real_votes)}/{len(results)}",
                'method': 'weighted_average'
            }
        else:
            consensus = None
        
        return {
            'success': True,
            'ensemble': True,
            'models': results,
            'consensus': consensus
        }


# ===============================
# INITIALIZE MODEL MANAGER
# ===============================
device = torch.device('cuda' if torch.cuda.is_available() else 
                     'mps' if torch.backends.mps.is_available() else 'cpu')

model_manager = ModelManager(device)


# ===============================
# UTILITY FUNCTIONS
# ===============================
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ===============================
# FLASK ROUTES
# ===============================
@app.route('/')
def index():
    """Serve the main HTML page"""
    return app.send_static_file('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle image upload and prediction
    Supports: Single model or Ensemble mode
    """
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: JPG, PNG, GIF, WEBP'}), 400
        
        # Get parameters
        model_type = request.form.get('model_type', 'auto')
        ensemble = request.form.get('ensemble', 'false').lower() == 'true'
        explain = request.form.get('explain', 'true').lower() == 'true'  # Explanation enabled by default
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load image
        image = Image.open(filepath).convert('RGB')
        
        # Make prediction
        if ensemble:
            result = model_manager.predict_ensemble(image, generate_explanation=explain)
        else:
            result = model_manager.predict_single(image, model_type, generate_explanation=explain)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    models_loaded = sum([
        model_manager.facenet_model is not None,
        model_manager.buildings_model is not None,
        model_manager.nature_model is not None
    ])
    
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'models_available': {
            'facenet': model_manager.facenet_model is not None,
            'buildings': model_manager.buildings_model is not None,
            'nature': model_manager.nature_model is not None
        },
        'device': str(device)
    })


@app.route('/models', methods=['GET'])
def get_models():
    """Get available models info"""
    return jsonify({
        'available_models': [
            {
                'id': 'face',
                'name': 'FaceNet (VGGFace2)',
                'type': 'Face Detection',
                'status': 'loaded' if model_manager.facenet_model is not None else 'not_loaded'
            },
            {
                'id': 'building',
                'name': 'Buildings Detector',
                'type': 'Building Detection',
                'status': 'loaded' if model_manager.buildings_model is not None else 'not_loaded'
            },
            {
                'id': 'nature',
                'name': 'Nature Detector',
                'type': 'Nature Detection',
                'status': 'loaded' if model_manager.nature_model is not None else 'not_loaded'
            }
        ]
    })


# ===============================
# RUN APPLICATION
# ===============================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("AI VISION - MULTI-MODEL DETECTION SYSTEM")
    print("="*60)
    print(f"  Server: http://localhost:5001")
    print(f"  Device: {device}")
    print(f"  Models: FaceNet + Buildings + Nature")
    print(f"  Features: Auto-detect, Manual, Ensemble")
    print("="*60 + "\n")
    
    # Run on port 5001
    app.run(debug=True, host='0.0.0.0', port=5001)
