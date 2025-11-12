"""
Flask Backend for FaceNet Face Deepfake Detection
Connects the trained FaceNet model with the web UI
"""

import os
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from facenet_pytorch import MTCNN

# Import model configuration
import config_facenet as config
from model_facenet import FaceNetClassifier

app = Flask(__name__, 
            static_folder='gui',
            static_url_path='')

# Configuration
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'jfif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the trained FaceNet model
MODEL_PATH = config.MODEL_DIR / 'best_model_facenet.pth'
print(f"Loading FaceNet model from: {MODEL_PATH}")

device = config.DEVICE

# Load checkpoint first to get config
checkpoint = torch.load(MODEL_PATH, map_location=device)
model_config = checkpoint.get('config', {})

# Create model with same config as training
model = FaceNetClassifier(
    num_classes=model_config.get('num_classes', 1),
    pretrained=model_config.get('pretrained', 'vggface2'),
    dropout_rate=config.DROPOUT_RATE
).to(device)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ FaceNet model loaded successfully on {device}")
print(f"  Validation accuracy: {checkpoint.get('accuracy', 'N/A')}")
if isinstance(checkpoint.get('accuracy'), float):
    print(f"  Validation accuracy: {checkpoint['accuracy']*100:.2f}%")
print(f"  Pretrained on: {model_config.get('pretrained', 'VGGFace2')}")

# Initialize MTCNN for face detection
mtcnn = MTCNN(
    image_size=config.IMG_SIZE,
    margin=config.FACE_DETECTION_MARGIN,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.9],
    keep_all=False,
    device=device
)
print(f" MTCNN face detector initialized")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                       std=[0.5, 0.5, 0.5])
])

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_face(image):
    """
    Detect face using MTCNN
    Returns: cropped face PIL Image or None
    """
    try:
        # MTCNN returns face tensor
        face = mtcnn(image)
        
        if face is not None:
            # Convert tensor to PIL Image
            face = face.permute(1, 2, 0).cpu().numpy()
            face = ((face * 128) + 128).clip(0, 255).astype('uint8')
            face = Image.fromarray(face)
            return face
        else:
            return None
    except Exception as e:
        print(f"Face detection error: {e}")
        return None

def predict_image(image_path):
    """
    Predict if a face image is Real or Fake using FaceNet
    Returns: (prediction, confidence, face_detected)
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Detect face
        face = detect_face(image)
        face_detected = face is not None
        
        if not face_detected:
            # No face detected - use whole image
            face = image.resize((config.IMG_SIZE, config.IMG_SIZE))
        
        # Preprocess
        image_tensor = transform(face).unsqueeze(0).to(device)
        
        # Make prediction
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
        
        return prediction, confidence_str, probability, face_detected
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise e

@app.route('/')
def index():
    """Serve the main HTML page"""
    return app.send_static_file('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS)"""
    return app.send_static_file(path)

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle image upload and prediction
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
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        prediction, confidence_str, probability, face_detected = predict_image(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Return result
        result = {
            'prediction': prediction,
            'confidence': confidence_str,
            'probability': round(probability, 4),
            'face_detected': face_detected,
            'model': 'FaceNet (VGGFace2)',
            'success': True
        }
        
        if not face_detected:
            result['warning'] = 'No face detected - analyzed full image'
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': str(device),
        'model_path': str(MODEL_PATH),
        'model_type': 'FaceNet',
        'face_detection': 'MTCNN'
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" FACENET FACE DEEPFAKE DETECTION - Starting Server")
    print("="*60)
    print(f" Server: http://localhost:5001")
    print(f" Model: FaceNet (InceptionResnetV1)")
    print(f" Face Detection: MTCNN")
    print(f" Device: {device}")
    print(f" Image Size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print("="*60 + "\n")
    
    # Run on port 5001
    app.run(debug=True, host='0.0.0.0', port=5001)
