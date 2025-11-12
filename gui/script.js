// Futuristic AI Vision JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Check if jsPDF is loaded
    console.log('=== jsPDF LIBRARY CHECK ===');
    console.log('window.jsPDF:', window.jsPDF);
    console.log('typeof window.jsPDF:', typeof window.jsPDF);
    if (window.jsPDF) {
        console.log('✓ jsPDF library loaded successfully');
        if (window.jsPDF.jsPDF) {
            console.log('✓ jsPDF constructor available at window.jsPDF.jsPDF');
        }
    } else {
        console.error('✗ jsPDF library NOT loaded');
    }
    
    // Initialize splash screen
    initSplashScreen();
    
    // Initialize main app functionality
    setTimeout(() => {
        initMainApp();
    }, 5000); // Show splash for 5 seconds
});

// Splash Screen Animation
function initSplashScreen() {
    const progressFill = document.getElementById('progressFill');
    const loadingPercentage = document.getElementById('loadingPercentage');
    const statusLines = document.querySelectorAll('.status-line');
    
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15 + 5; // Random progress increments
        if (progress > 100) progress = 100;
        
        progressFill.style.width = progress + '%';
        loadingPercentage.textContent = Math.floor(progress) + '%';
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            setTimeout(() => {
                hideSplashScreen();
            }, 1000);
        }
    }, 200);
    
    // Animate status lines
    statusLines.forEach((line, index) => {
        setTimeout(() => {
            line.style.color = '#00ff88';
            line.innerHTML = '✓ ' + line.innerHTML.substring(2);
        }, (index + 1) * 1000);
    });
}

// Hide splash screen and show main app
function hideSplashScreen() {
    const splashScreen = document.getElementById('splashScreen');
    const mainApp = document.getElementById('mainApp');
    
    splashScreen.style.opacity = '0';
    splashScreen.style.transform = 'scale(0.8)';
    splashScreen.style.transition = 'all 0.8s ease';
    
    setTimeout(() => {
        splashScreen.style.display = 'none';
        mainApp.style.display = 'flex';
        mainApp.style.opacity = '0';
        mainApp.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            mainApp.style.transition = 'all 0.8s ease';
            mainApp.style.opacity = '1';
            mainApp.style.transform = 'translateY(0)';
        }, 100);
    }, 800);
}

// Initialize main application functionality
function initMainApp() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('imageFile');
    const browseLink = document.getElementById('browseLink');
    const previewSection = document.getElementById('preview');
    const previewImage = document.getElementById('previewImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const removeBtn = document.getElementById('removeImage');
    const loadingSection = document.getElementById('loading');
    const resultSection = document.getElementById('result');
    
    // File input and drag-drop functionality
    browseLink.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);
    
    // Remove image button
    removeBtn.addEventListener('click', removeImage);
    
    // Action buttons
    document.getElementById('analyzeAnother')?.addEventListener('click', () => {
        removeImage();
        resultSection.style.display = 'none';
    });
    
    document.getElementById('downloadReport')?.addEventListener('click', downloadReport);
    
    // Ensemble action buttons
    document.getElementById('analyzeAnotherEnsemble')?.addEventListener('click', () => {
        removeImage();
        document.getElementById('ensembleResult').style.display = 'none';
    });
    
    document.getElementById('downloadReportEnsemble')?.addEventListener('click', downloadReport);
    
    // Add hover effects
    addHoverEffects();
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// File selection handler
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Handle file processing
function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showNotification('Please select a valid image file', 'error');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        showPreview(e.target.result);
    };
    reader.readAsDataURL(file);
}

// Show image preview
function showPreview(imageSrc) {
    const previewSection = document.getElementById('preview');
    const previewImage = document.getElementById('previewImage');
    const uploadSection = document.querySelector('.upload-section');
    const featuresSection = document.getElementById('features');
    
    previewImage.src = imageSrc;
    lastImageData = imageSrc; // Store for PDF generation
    previewSection.style.display = 'block';
    uploadSection.style.display = 'none';
    
    // Hide features section when image is uploaded
    if (featuresSection) {
        featuresSection.style.display = 'none';
    }
    
    // Trigger scan animation
    setTimeout(() => {
        const scanEffect = document.querySelector('.scan-effect');
        if (scanEffect) {
            scanEffect.style.animation = 'scanEffect 2s infinite';
        }
    }, 500);
}

// Remove image and reset
function removeImage() {
    const previewSection = document.getElementById('preview');
    const uploadSection = document.querySelector('.upload-section');
    const fileInput = document.getElementById('imageFile');
    const loadingSection = document.getElementById('loading');
    const resultSection = document.getElementById('result');
    const ensembleSection = document.getElementById('ensembleResult');
    const featuresSection = document.getElementById('features');
    
    previewSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultSection.style.display = 'none';
    ensembleSection.style.display = 'none';
    uploadSection.style.display = 'block';
    
    // Show features section again when image is removed
    if (featuresSection) {
        featuresSection.style.display = 'block';
    }
    
    fileInput.value = '';
}

// Analyze image with AI
async function analyzeImage() {
    const fileInput = document.getElementById('imageFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showNotification('Please select an image first', 'error');
        return;
    }
    
    // Get user selections
    const modelType = document.getElementById('modelType').value;
    const ensembleMode = document.getElementById('ensembleMode').checked;
    
    showLoading(ensembleMode);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', modelType);
    formData.append('ensemble', ensembleMode.toString());
    formData.append('explain', 'true'); // Always generate explanations
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        // Check if response is JSON
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('Server error: Expected JSON response but got HTML. Please make sure the Flask server is running on port 5001.');
        }
        
        const result = await response.json();
        
        if (response.ok) {
            if (result.ensemble) {
                showEnsembleResults(result);
            } else {
                showResult(result);
            }
        } else {
            throw new Error(result.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error: ' + error.message, 'error');
        hideLoading();
    }
}

// Show loading animation
function showLoading(isEnsemble = false) {
    const previewSection = document.getElementById('preview');
    const loadingSection = document.getElementById('loading');
    const loadingSubtitle = loadingSection.querySelector('.loading-subtitle');
    
    previewSection.style.display = 'none';
    loadingSection.style.display = 'block';
    
    // Update message for ensemble mode
    if (isEnsemble) {
        loadingSubtitle.textContent = 'Running all 3 models for comprehensive analysis...';
    } else {
        loadingSubtitle.textContent = 'Deep neural networks are examining your image...';
    }
    
    // Animate analysis steps
    const steps = document.querySelectorAll('.step');
    steps.forEach((step, index) => {
        setTimeout(() => {
            steps.forEach(s => s.classList.remove('active'));
            step.classList.add('active');
        }, index * 1500);
    });
}

// Hide loading animation
function hideLoading() {
    const loadingSection = document.getElementById('loading');
    const previewSection = document.getElementById('preview');
    
    loadingSection.style.display = 'none';
    previewSection.style.display = 'block';
}

// Show analysis result
function showResult(data) {
    // Store result for download
    lastAnalysisResult = data;
    
    const loadingSection = document.getElementById('loading');
    const resultSection = document.getElementById('result');
    const ensembleSection = document.getElementById('ensembleResult');
    const resultBadge = document.getElementById('resultBadge');
    const resultTitle = document.getElementById('resultTitle');
    const resultDescription = document.getElementById('resultDescription');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    
    loadingSection.style.display = 'none';
    resultSection.style.display = 'block';
    ensembleSection.style.display = 'none';
    
    // Debug: Log the received data
    console.log('Received analysis data:', data);
    console.log('Prediction:', data.prediction);
    console.log('Model used:', data.model_used);
    console.log('Model name:', data.model_name);
    console.log('Confidence:', data.confidence);
    
    // Update model used indicator
    const modelIcon = document.getElementById('modelIcon');
    const modelText = document.getElementById('modelText');
    
    const modelIcons = {
        'face': '<i class="fas fa-user"></i>',
        'building': '<i class="fas fa-building"></i>',
        'nature': '<i class="fas fa-tree"></i>',
        'auto': '<i class="fas fa-magic"></i>'
    };
    
    const modelNames = {
        'face': 'FaceNet',
        'building': 'Buildings Detector',
        'nature': 'Nature Detector',
        'auto': 'Auto-Detect',
        'undefined': 'AI Model'
    };
    
    const modelUsed = data.model_used || data.model_type || 'auto';
    modelIcon.innerHTML = modelIcons[modelUsed] || modelIcons['auto'] || '<i class="fas fa-robot"></i>';
    
    // Get model name with fallback
    const modelName = modelNames[modelUsed] || data.model_name || 'AI Model';
    
    if (data.auto_detected) {
        modelText.textContent = `Analyzed by: ${modelName} (Auto-detected)`;
    } else {
        modelText.textContent = `Analyzed by: ${modelName}`;
    }
    
    // Add face detection indicator
    if (data.face_detected) {
        modelText.textContent += ' ✓ Face detected';
    }
    
    // Determine result type
    const prediction = data.prediction || 'UNKNOWN';
    const isReal = prediction === 'REAL';
    
    // Extract confidence percentage from string (e.g., "67.3% Real" -> 67.3)
    let confidence;
    if (typeof data.confidence === 'string') {
        // Extract number from string like "67.3% Real" or "32.7% Fake"
        const match = data.confidence.match(/(\d+\.?\d*)%?/);
        confidence = match ? parseFloat(match[1]) : 50.0;
    } else if (typeof data.confidence === 'number') {
        // If it's a decimal (0-1), convert to percentage
        if (data.confidence <= 1) {
            confidence = data.confidence * 100;
        } else {
            confidence = data.confidence;
        }
    } else {
        confidence = 50.0;
    }
    
    // Ensure confidence is between 0 and 100
    confidence = Math.min(100, Math.max(0, confidence));
    
    // Round to 1 decimal place
    confidence = Math.round(confidence * 10) / 10;
    
    // Ensure confidence is valid
    if (isNaN(confidence) || !isFinite(confidence)) {
        confidence = 50.0;
    }
    
    // Update badge with animation delay
    setTimeout(() => {
        resultBadge.className = `result-badge ${isReal ? 'real' : 'fake'}`;
        resultBadge.innerHTML = isReal ? '<i class="fas fa-check-circle"></i> ' + prediction : '<i class="fas fa-exclamation-triangle"></i> ' + prediction;
    }, 200);
    
    // Update title with animation delay
    setTimeout(() => {
        resultTitle.textContent = isReal ? 'Authentic Image' : 'AI Generated Image';
        resultTitle.style.color = isReal ? '#00ff88' : '#ff4757';
    }, 400);
    
    // Update description with animation delay
    setTimeout(() => {
        resultDescription.innerHTML = isReal ? 
            'Our AI analysis indicates this image appears to be <strong>authentic</strong> and likely captured by a camera or created through traditional means.' :
            'Our AI analysis indicates this image appears to be <strong>artificially generated</strong> using AI or deep learning techniques.';
    }, 600);
    
    // Animate confidence bar
    setTimeout(() => {
        confidenceFill.style.width = '0%';
        const color = isReal ? '#00ff88' : '#ff4757';
        confidenceFill.style.background = isReal ? 
            'linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)' : 
            'linear-gradient(135deg, #ff4757 0%, #c44569 100%)';
        confidenceFill.style.color = color;
        
        setTimeout(() => {
            confidenceFill.style.width = confidence + '%';
        }, 100);
        
        // Animate confidence text
        animateValue(confidenceText, 0, confidence, 1500, '%', color);
    }, 800);
    
    // Add some sparkle effects
    setTimeout(() => {
        createSparkleEffect(resultSection);
    }, 1000);
    
    // Display explanation if available
    if (data.explanation) {
        displayExplanation(data.explanation, 'single');
    }
}

// Display Grad-CAM explanation
function displayExplanation(explanation, mode = 'single') {
    const explanationSection = document.getElementById('explanationSection');
    const heatmapImage = document.getElementById('heatmapImage');
    const explanationText = document.getElementById('explanationText');
    
    if (!explanationSection) return;
    
    explanationSection.style.display = 'block';
    
    // Display heat map if available
    if (explanation.heatmap) {
        heatmapImage.src = explanation.heatmap;
        heatmapImage.style.display = 'block';
    } else {
        heatmapImage.style.display = 'none';
    }
    
    // Display explanation text
    if (explanation.text) {
        explanationText.innerHTML = `
            <div class="explanation-header">
                <span class="explanation-icon">🔍</span>
                <span class="explanation-title">Why this prediction?</span>
            </div>
            <div class="explanation-content">
                ${explanation.text}
            </div>
        `;
    }
    
    // Animate in
    setTimeout(() => {
        explanationSection.style.opacity = '1';
        explanationSection.style.transform = 'translateY(0)';
    }, 100);
}

// Show ensemble results (all 3 models)
function showEnsembleResults(data) {
    // Store result for download
    lastAnalysisResult = data;
    
    const loadingSection = document.getElementById('loading');
    const resultSection = document.getElementById('result');
    const ensembleSection = document.getElementById('ensembleResult');
    
    loadingSection.style.display = 'none';
    resultSection.style.display = 'none';
    ensembleSection.style.display = 'block';
    
    // Display each model's result
    const models = data.models;
    
    // FaceNet
    if (models.face) {
        updateEnsembleModel('Face', models.face);
    }
    
    // Buildings
    if (models.building) {
        updateEnsembleModel('Building', models.building);
    }
    
    // Nature
    if (models.nature) {
        updateEnsembleModel('Nature', models.nature);
    }
    
    // Display consensus
    if (data.consensus) {
        const consensus = data.consensus;
        const isReal = consensus.prediction === 'REAL';
        
        const consensusBadge = document.getElementById('consensusBadge');
        const consensusPrediction = document.getElementById('consensusPrediction');
        const consensusConfidence = document.getElementById('consensusConfidence');
        const consensusAgreement = document.getElementById('consensusAgreement');
        
        // Update badge
        consensusBadge.className = `consensus-badge ${isReal ? 'real' : 'fake'}`;
        consensusBadge.innerHTML = isReal ? '<i class="fas fa-check-circle"></i> ' + consensus.prediction : '<i class="fas fa-exclamation-triangle"></i> ' + consensus.prediction;
        
        // Update prediction
        consensusPrediction.textContent = isReal ? 'Authentic Image' : 'AI-Generated Image';
        
        // Extract confidence
        let confidence = consensus.confidence;
        if (confidence <= 1) {
            confidence = confidence * 100;
        }
        confidence = Math.round(confidence * 10) / 10;
        
        // Update details
        consensusConfidence.textContent = `Combined Confidence: ${confidence}%`;
        consensusAgreement.textContent = `Agreement: ${consensus.agreement} models`;
    }
    
    // Add sparkle effects
    setTimeout(() => {
        createSparkleEffect(ensembleSection);
    }, 1000);
    
    // Display ensemble explanations if available
    displayEnsembleExplanations(models);
}

// Display explanations for ensemble mode
function displayEnsembleExplanations(models) {
    const ensembleExplanationSection = document.getElementById('ensembleExplanationSection');
    if (!ensembleExplanationSection) return;
    
    let hasExplanations = false;
    let explanationsHTML = '<div class="ensemble-explanations">';
    
    // Check each model for explanations
    const modelInfo = {
        'face': { name: 'FaceNet', icon: '👤' },
        'building': { name: 'Buildings', icon: '🏢' },
        'nature': { name: 'Nature', icon: '🌲' }
    };
    
    for (const [modelType, modelData] of Object.entries(models)) {
        if (modelData.explanation) {
            hasExplanations = true;
            const info = modelInfo[modelType];
            
            explanationsHTML += `
                <div class="ensemble-explanation-card">
                    <div class="explanation-card-header">
                        <span class="model-icon">${info.icon}</span>
                        <span class="model-name">${info.name} Model</span>
                    </div>
                    ${modelData.explanation.heatmap ? `
                    <div class="explanation-heatmap">
                        <img src="${modelData.explanation.heatmap}" alt="${info.name} Heat Map" />
                    </div>
                    ` : ''}
                    <div class="explanation-text-content">
                        ${modelData.explanation.text}
                    </div>
                </div>
            `;
        }
    }
    
    explanationsHTML += '</div>';
    
    if (hasExplanations) {
        ensembleExplanationSection.innerHTML = `
            <div class="explanation-section-header">
                <span class="explanation-icon">🔍</span>
                <span class="explanation-title">Visual Explanations</span>
            </div>
            ${explanationsHTML}
        `;
        ensembleExplanationSection.style.display = 'block';
    } else {
        ensembleExplanationSection.style.display = 'none';
    }
}

// Update individual ensemble model result
function updateEnsembleModel(modelType, modelData) {
    const modelId = modelType.toLowerCase();
    const bar = document.getElementById(`ensemble${modelType}Bar`);
    const text = document.getElementById(`ensemble${modelType}Text`);
    
    if (!bar || !text) return;
    
    const isReal = modelData.prediction === 'REAL';
    let confidence = modelData.confidence;
    
    // Convert to percentage if needed
    if (confidence <= 1) {
        confidence = confidence * 100;
    }
    confidence = Math.round(confidence * 10) / 10;
    
    // Update bar
    setTimeout(() => {
        bar.style.width = '0%';
        bar.className = `model-bar ${isReal ? 'real' : 'fake'}`;
        
        setTimeout(() => {
            bar.style.width = confidence + '%';
        }, 100);
    }, 500);
    
    // Update text
    setTimeout(() => {
        text.innerHTML = `
            <span class="prediction-label">${modelData.prediction}</span>
            <span class="prediction-confidence">${confidence}%</span>
        `;
    }, 700);
}

// Animate number counting
function animateValue(element, start, end, duration, suffix = '', color = '#00d4ff') {
    const range = end - start;
    const increment = range / (duration / 16); // 60fps
    let current = start;
    
    element.style.color = color;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = current.toFixed(1) + suffix;
    }, 16);
}

// Global variable to store last analysis result
let lastAnalysisResult = null;
let lastImageData = null;

// Download report as PDF
function downloadReport() {
    console.log('=== DOWNLOAD REPORT ===');
    console.log('lastAnalysisResult:', lastAnalysisResult);
    
    if (!lastAnalysisResult) {
        alert('No analysis result available. Please analyze an image first.');
        return;
    }
    
    try {
        generatePDFReport();
    } catch (error) {
        console.error('PDF generation error:', error);
        alert('Failed to generate PDF: ' + error.message);
    }
}

// Generate PDF report
function generatePDFReport() {
    console.log('=== GENERATING PDF ===');
    
    // Handle both UMD and global loading patterns
    const jsPDF = window.jspdf?.jsPDF || window.jsPDF?.jsPDF || window.jsPDF;
    
    if (!jsPDF) {
        throw new Error('jsPDF library not found. Please refresh the page.');
    }
    
    const doc = new jsPDF();
    
    const result = lastAnalysisResult;
    const prediction = result.prediction || 'FAKE';
    const isReal = prediction === 'REAL';
    
    // Extract confidence
    let confidence = 50;
    if (typeof result.confidence === 'string') {
        const match = result.confidence.match(/(\d+\.?\d*)/);
        if (match) confidence = parseFloat(match[1]);
    } else if (typeof result.confidence === 'number') {
        confidence = result.confidence <= 1 ? result.confidence * 100 : result.confidence;
    }
    confidence = Math.round(confidence * 10) / 10;
    
    // Get model info
    const modelUsed = result.model_used || result.model_type || 'auto';
    const modelNames = {
        'face': 'FaceNet Model',
        'building': 'Buildings Detection Model', 
        'nature': 'Nature Detection Model',
        'auto': 'Auto-Detection'
    };
    const modelName = modelNames[modelUsed] || 'AI Detection Model';
    
    console.log('PDF Data:', { prediction, isReal, confidence, modelName });
    
    const pageWidth = doc.internal.pageSize.width;
    const pageHeight = doc.internal.pageSize.height;
    let yPos = 20;
    
    // === TITLE SECTION ===
    doc.setFillColor(0, 102, 255);
    doc.rect(0, 0, pageWidth, 40, 'F');
    
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(28);
    doc.setFont('helvetica', 'bold');
    doc.text('AI VISION', pageWidth / 2, 20, { align: 'center' });
    
    doc.setFontSize(14);
    doc.setFont('helvetica', 'normal');
    doc.text('Image Authentication Report', pageWidth / 2, 32, { align: 'center' });
    
    yPos = 55;
    
    // Report metadata
    const now = new Date();
    const timestamp = now.toLocaleString();
    const reportId = now.getTime();
    
    doc.setFontSize(9);
    doc.setTextColor(120, 120, 120);
    doc.setFont('helvetica', 'normal');
    doc.text(`Report Date: ${timestamp}`, 15, yPos);
    doc.text(`Model: ${modelName}`, 15, yPos + 5);
    yPos += 15;
    
    // === IMAGE SECTION ===
    if (lastImageData) {
        try {
            doc.setFontSize(12);
            doc.setTextColor(0, 0, 0);
            doc.setFont('helvetica', 'bold');
            doc.text('Analyzed Image:', 15, yPos);
            yPos += 8;
            
            const imgWidth = 100;
            const imgHeight = 70;
            const imgX = (pageWidth - imgWidth) / 2;
            
            // Add border
            doc.setDrawColor(200, 200, 200);
            doc.setLineWidth(0.5);
            doc.rect(imgX - 1, yPos - 1, imgWidth + 2, imgHeight + 2);
            
            // Add image
            doc.addImage(lastImageData, 'JPEG', imgX, yPos, imgWidth, imgHeight);
            yPos += imgHeight + 15;
        } catch (e) {
            console.error('Could not add image to PDF:', e);
            yPos += 10;
        }
    } else {
        yPos += 5;
    }
    
    // === RESULT SECTION ===
    doc.setFontSize(12);
    doc.setTextColor(0, 0, 0);
    doc.setFont('helvetica', 'bold');
    doc.text('Analysis Result:', 15, yPos);
    yPos += 10;
    
    // Result badge
    const badgeText = prediction;
    doc.setFontSize(16);
    doc.setFont('helvetica', 'bold');
    
    if (isReal) {
        doc.setFillColor(16, 185, 129);
        doc.setTextColor(16, 185, 129);
    } else {
        doc.setFillColor(239, 68, 68);
        doc.setTextColor(239, 68, 68);
    }
    
    // Draw badge box
    const badgeX = (pageWidth - 60) / 2;
    doc.roundedRect(badgeX, yPos - 8, 60, 12, 2, 2, 'F');
    
    // Badge text
    doc.setTextColor(255, 255, 255);
    doc.text(badgeText, pageWidth / 2, yPos, { align: 'center' });
    yPos += 15;
    
    // Classification text
    doc.setFontSize(18);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(0, 0, 0);
    const statusText = isReal ? 'Authentic Image' : 'AI-Generated Image';
    doc.text(statusText, pageWidth / 2, yPos, { align: 'center' });
    yPos += 8;
    
    // Confidence
    doc.setFontSize(14);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(80, 80, 80);
    doc.text(`Confidence: ${confidence}%`, pageWidth / 2, yPos, { align: 'center' });
    yPos += 15;
    
    // === REASON SECTION ===
    // Reason text
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(60, 60, 60);
    
    const reasonText = isReal
        ? 'This image appears to be authentic based on our AI analysis. The neural network detected natural patterns, authentic camera characteristics, and absence of common AI generation artifacts. The image shows typical properties of real-world photography including natural lighting, organic textures, and genuine compression patterns.'
        : 'This image appears to be AI-generated based on our analysis. The system detected patterns commonly found in AI-generated content, such as synthetic characteristics, unusual texture patterns, or AI generation artifacts. These indicators suggest the image was created using deep learning models or generative AI systems.';
    
    const reasonLines = doc.splitTextToSize(reasonText, pageWidth - 30);
    doc.text(reasonLines, 15, yPos);
    yPos += (reasonLines.length * 5) + 10;
    
    // === FOOTER ===
    // Add some space before footer
    if (yPos < pageHeight - 30) {
        yPos = pageHeight - 30;
    }
    
    // Separator line
    doc.setDrawColor(200, 200, 200);
    doc.setLineWidth(0.5);
    doc.line(15, yPos, pageWidth - 15, yPos);
    yPos += 8;
    
    // Footer text
    doc.setFontSize(9);
    doc.setTextColor(120, 120, 120);
    doc.setFont('helvetica', 'normal');
    doc.text('Disclaimer: This analysis is AI-generated and should be used as reference only.', pageWidth / 2, yPos, { align: 'center' });
    yPos += 5;
    doc.text('© 2025 AI Vision System - Advanced Image Authentication', pageWidth / 2, yPos, { align: 'center' });
    
    // Save the PDF
    const fileName = `AI_Vision_Report_${now.getTime()}.pdf`;
    doc.save(fileName);
    console.log('PDF saved:', fileName);
}

// Text report removed - PDF only

// Create sparkle effect
function createSparkleEffect(container) {
    const colors = ['#00d4ff', '#00ff88', '#8a2be2', '#ff006e'];
    
    for (let i = 0; i < 20; i++) {
        setTimeout(() => {
            const sparkle = document.createElement('div');
            const size = Math.random() * 4 + 2;
            const color = colors[Math.floor(Math.random() * colors.length)];
            
            sparkle.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                background: ${color};
                border-radius: 50%;
                pointer-events: none;
                animation: sparkle 2s ease-out forwards;
                top: ${Math.random() * 100}%;
                left: ${Math.random() * 100}%;
                box-shadow: 0 0 10px ${color};
                z-index: 10;
            `;
            
            container.style.position = 'relative';
            container.appendChild(sparkle);
            
            setTimeout(() => sparkle.remove(), 2000);
        }, i * 100);
    }
}

// Add particle effects on hover
function addHoverEffects() {
    const uploadCard = document.getElementById('uploadArea');
    
    if (uploadCard) {
        uploadCard.addEventListener('mousemove', (e) => {
            const rect = uploadCard.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            if (Math.random() > 0.95) {
                const particle = document.createElement('div');
                particle.style.cssText = `
                    position: absolute;
                    width: 3px;
                    height: 3px;
                    background: rgba(0, 212, 255, 0.6);
                    border-radius: 50%;
                    pointer-events: none;
                    left: ${x}px;
                    top: ${y}px;
                    animation: particleFade 1s ease-out forwards;
                `;
                
                uploadCard.appendChild(particle);
                setTimeout(() => particle.remove(), 1000);
            }
        });
    }
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    // Choose icon and color based on type
    let icon, bgColor;
    switch(type) {
        case 'error':
            icon = '❌';
            bgColor = '#ff4757';
            break;
        case 'success':
            icon = '✅';
            bgColor = '#00ff88';
            break;
        default:
            icon = 'ℹ️';
            bgColor = '#00d4ff';
    }
    
    notification.innerHTML = `
        <div class="notification-content">
            <span class="notification-icon">${icon}</span>
            <span class="notification-text">${message}</span>
        </div>
    `;
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${bgColor};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        z-index: 10000;
        animation: slideInRight 0.5s ease;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        backdrop-filter: blur(20px);
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.5s ease forwards';
        setTimeout(() => notification.remove(), 500);
    }, 3000);
}

// Add sparkle animation styles
const sparkleStyles = document.createElement('style');
sparkleStyles.textContent = `
    @keyframes sparkle {
        0% {
            opacity: 0;
            transform: scale(0) rotate(0deg);
        }
        50% {
            opacity: 1;
            transform: scale(1) rotate(180deg);
        }
        100% {
            opacity: 0;
            transform: scale(0) rotate(360deg);
        }
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
`;
document.head.appendChild(sparkleStyles);