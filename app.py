"""
Flask Web Application for EKG Diagnosis using a Trained PyTorch Model

This application provides a web interface for real-time EKG diagnosis using
a deep learning model that combines CNN, LSTM, and attention mechanisms.
The model analyzes 12-lead EKG data to detect various cardiac conditions.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import torch.nn as nn
import os
import traceback
import json
import logging
from preprocess import normalize_signal, resample_signal  # Import custom preprocessing functions

# Configure application logging for debugging and monitoring
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the Flask application instance
app = Flask(__name__)

# ------------------------------------------------------
# Neural Network Architecture Definition
# (Must exactly match the architecture used during training)
# ------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Implements a temporal attention mechanism that learns to focus on 
    important time steps in the LSTM output sequence.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        # Linear layer that computes attention scores for each time step
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        """
        Forward pass for temporal attention.
        
        Args:
            lstm_output: Output from LSTM layer with shape (batch_size, seq_length, hidden_dim)
            
        Returns:
            context: Weighted sum of LSTM outputs (batch_size, hidden_dim)
            weights: Attention weights for each time step (batch_size, seq_length, 1)
        """
        # Compute attention scores for each time step
        scores = self.attention(lstm_output)
        # Convert scores to probabilities using softmax
        weights = torch.softmax(scores, dim=1)
        # Compute context vector as weighted sum of LSTM outputs
        context = torch.sum(weights * lstm_output, dim=1)
        return context, weights


class CNN_LSTM_Attention(nn.Module):
    """
    Combined CNN-LSTM model with attention mechanism for EKG signal classification.
    
    Architecture:
    1. 1D CNN for local feature extraction from EKG signals
    2. LSTM for capturing temporal dependencies
    3. Attention mechanism to focus on clinically relevant time segments
    4. Fully connected layer for final classification
    """
    def __init__(self, input_channels=12, num_classes=5, lstm_hidden=128):
        super(CNN_LSTM_Attention, self).__init__()
        
        # CNN layers for feature extraction
        self.cnn = nn.Sequential(
            # First convolutional layer: 12 input channels -> 32 output channels
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            # Second convolutional layer: 32 input channels -> 64 output channels
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        # Bidirectional LSTM for capturing temporal patterns in both directions
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden, 
                           batch_first=True, bidirectional=True)
        
        # Attention mechanism to weight important time steps
        self.attention = TemporalAttention(lstm_hidden * 2)  # x2 for bidirectional
        
        # Final classification layer
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        """
        Forward pass through the complete network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_channels)
            
        Returns:
            out: Model predictions (batch_size, num_classes)
        """
        # Permute for CNN: (batch, channels, time) format
        x = x.permute(0, 2, 1)
        # Extract features using CNN
        x = self.cnn(x)
        # Permute back for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1)
        # Process temporal patterns with LSTM
        lstm_out, _ = self.lstm(x)
        # Apply attention to focus on important time steps
        context, attn_weights = self.attention(lstm_out)
        # Final classification
        out = self.fc(context)
        return out


# ------------------------------------------------------
# Model Loading and Initialization
# ------------------------------------------------------

# Global variable to hold the loaded model
model = None

# File path to the trained model weights
model_path = r"C:\Users\15166\OneDrive\Desktop\final_ekg\models\cnn_lstm_attention_5diagnoses.pth"

# Comprehensive model loading debug information
logger.debug("=" * 50)
logger.debug("MODEL LOADING DEBUG INFORMATION")
logger.debug("=" * 50)
logger.debug(f"Model path: {model_path}")
logger.debug(f"File exists: {os.path.exists(model_path)}")

# Check file existence and size for debugging
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path)
    logger.debug(f"File size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
else:
    logger.error(f"Model file not found at: {model_path}")

try:
    # Validate model file existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    
    # Initialize model with the correct architecture parameters
    model = CNN_LSTM_Attention(input_channels=12, num_classes=5)
    
    # Determine available hardware (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f"Using device: {device}")
    
    # Load trained weights into the model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move model to appropriate device and set to evaluation mode
    model.to(device)
    model.eval()  # Disables dropout and batch normalization during inference
    logger.info("Model loaded successfully!")
    
except Exception as e:
    # Comprehensive error handling for model loading failures
    logger.error(f"Error loading model: {e}")
    logger.error(traceback.format_exc())
    model = None  # Ensure model is None if loading fails

# ------------------------------------------------------
# Diagnosis Class Definitions
# ------------------------------------------------------

# Human-readable class names corresponding to model outputs
class_names = ["NORM", "MI", "STTC", "CD", "HYP"]
# NORM: Normal rhythm
# MI: Myocardial Infarction (heart attack)
# STTC: ST-T Wave Changes
# CD: Conduction Disorder
# HYP: Hypertrophy

# ------------------------------------------------------
# Data Preprocessing Helper Functions
# ------------------------------------------------------

def preprocess_ekg_data(ekg_data, target_length=1000):
    """
    Preprocess raw EKG data to match model input requirements.
    
    Processing steps:
    1. Convert to numpy array
    2. Handle lead count mismatch (3-lead to 12-lead conversion)
    3. Resample to target length
    4. Normalize signal values
    5. Convert to PyTorch tensor with appropriate dimensions
    
    Args:
        ekg_data: List of 3-lead EKG measurements [[v1, v2, v3], ...]
        target_length: Desired length of the output sequence (default: 1000)
        
    Returns:
        ekg_tensor: Preprocessed tensor ready for model inference
    """
    # Convert input data to numpy array with float32 precision
    ekg_array = np.array(ekg_data, dtype=np.float32)
    
    # Handle 3-lead to 12-lead conversion (common in mobile EKG devices)
    if ekg_array.shape[1] == 3:
        # Duplicate the 3 leads to simulate 12 leads
        # Note: In production, this should be replaced with proper lead derivation
        ekg_array = np.tile(ekg_array, (1, 4))[:, :12]
    
    # Resample signal to consistent length (required for fixed-size model input)
    ekg_resampled = resample_signal(ekg_array, target_length)
    
    # Normalize signal to standard range (typically 0-1 or z-score)
    ekg_normalized = normalize_signal(ekg_resampled)
    
    # Convert to PyTorch tensor and add batch dimension
    ekg_tensor = torch.tensor(ekg_normalized, dtype=torch.float32)
    ekg_tensor = ekg_tensor.unsqueeze(0)  # Add batch dimension (batch_size=1)
    
    return ekg_tensor

# ------------------------------------------------------
# Flask Route Definitions
# ------------------------------------------------------

@app.route("/")
def index():
    """Serve the main application interface."""
    return render_template("index.html")

@app.route("/health")
def health_check():
    """
    Health check endpoint for monitoring and deployment validation.
    
    Returns:
        JSON response with application status and model information
    """
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": model_path,
        "file_exists": os.path.exists(model_path)
    })

@app.post("/diagnose")
def diagnose():
    """
    Main diagnosis endpoint that processes EKG data and returns predictions.
    
    Expected JSON input:
    {
        "ekg_data": [[v1, v2, v3], [v1, v2, v3], ...]  # List of 3-lead measurements
    }
    
    Returns:
        JSON response with diagnosis results or error message
    """
    logger.debug("Received diagnosis request at /diagnose endpoint")
    
    # Validate model availability
    if model is None:
        error_msg = "Model not loaded. Please check the server logs for details."
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500  # Internal Server Error

    try:
        # Parse JSON request data (fail silently if not JSON)
        data = request.get_json(silent=True) or {}
        logger.debug(f"Received data keys: {list(data.keys())}")
        
        # Validate presence of required EKG data
        if "ekg_data" not in data:
            error_msg = "No EKG data received in request"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400  # Bad Request

        # Extract and log raw data characteristics
        ekg_data = data["ekg_data"]
        logger.debug(f"Raw data shape: {np.array(ekg_data).shape if isinstance(ekg_data, list) else 'Not list'}")
        
        # Preprocess data for model consumption
        X_tensor = preprocess_ekg_data(ekg_data)
        logger.debug(f"Processed tensor shape: {X_tensor.shape}")

        # Perform model inference
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model(X_tensor)
            # Use sigmoid activation for multi-label classification probabilities
            probs = torch.sigmoid(outputs).numpy()[0]

        # Determine primary diagnosis (class with highest probability)
        diagnosis_idx = np.argmax(probs)
        diagnosis = class_names[diagnosis_idx] if diagnosis_idx < len(class_names) else f"class_{diagnosis_idx}"

        # Map diagnosis codes to human-readable messages
        diagnosis_messages = {
            "NORM": "Normal EKG - No significant abnormalities detected",
            "MI": "Myocardial Infarction (Heart Attack) detected - Urgent medical attention required",
            "STTC": "ST-T Wave Changes detected - May indicate ischemia or other conditions",
            "CD": "Conduction Disorder detected - Further evaluation recommended",
            "HYP": "Hypertrophy detected - May indicate chamber enlargement"
        }
        
        # Get appropriate message or default for unknown diagnoses
        diagnosis_message = diagnosis_messages.get(diagnosis, f"Diagnosis: {diagnosis}")

        # Construct response payload
        response = {
            "diagnosis": diagnosis,
            "message": diagnosis_message
            # Note: Probabilities intentionally excluded from response
        }
        
        logger.debug(f"Diagnosis results: {response}")
        return jsonify(response)

    except Exception as e:
        # Comprehensive error handling for diagnosis failures
        error_msg = f"Diagnosis processing error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())  # Log full traceback for debugging
        return jsonify({"error": error_msg}), 500  # Internal Server Error

# ------------------------------------------------------
# Application Entry Point
# ------------------------------------------------------

if __name__ == "__main__":
    """
    Main application entry point when run directly.
    
    Note: In production, consider using a production WSGI server like Gunicorn
    instead of Flask's built-in development server.
    """
    print("Starting Flask EKG Diagnosis Server...")
    print(f"Model loaded: {model is not None}")
    
    # Start Flask development server
    # debug=True enables auto-reload and detailed error pages
    # host='0.0.0.0' makes server accessible from other devices on network
    app.run(debug=True, host='0.0.0.0', port=5000)