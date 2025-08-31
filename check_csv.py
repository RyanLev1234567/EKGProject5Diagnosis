"""
Configuration and Path Settings for EKG Analysis Project

This module centralizes all file paths, model parameters, and configuration settings
to ensure consistency across the entire EKG analysis pipeline.
"""

import os
import torch
import json

# ------------------------------------------------------------------------------
# Directory Path Configuration
# ------------------------------------------------------------------------------

# Root directory of the project - contains all data, code, and results
BASE_DIR = r"C:\Users\15166\OneDrive\Desktop\final_ekg"

# Directory containing raw EKG record files (likely in WFDB format)
RECORDS_DIR = os.path.join(BASE_DIR, "records100")

# Directory for preprocessed and prepared data ready for model training
PREPARED_DIR = os.path.join(BASE_DIR, "prepared")

# Directory for saving trained model weights and architectures
MODELS_DIR = os.path.join(BASE_DIR, "models")
# Create models directory if it doesn't exist (prevents errors during model saving)
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Data File Paths
# ------------------------------------------------------------------------------

# PTB-XL database CSV file containing metadata and diagnostic information
PTBXL_CSV = os.path.join(BASE_DIR, "ptbxl_database.csv")

# SCP statements CSV file containing standardized diagnostic codes and descriptions
SCP_CSV = os.path.join(BASE_DIR, "scp_statements.csv")

# ------------------------------------------------------------------------------
# Model Training Hyperparameters
# ------------------------------------------------------------------------------

# Number of samples processed before model parameters are updated
BATCH_SIZE = 32  # Balanced between memory usage and training stability

# Step size for parameter updates during optimization
LEARNING_RATE = 1e-3  # Common starting point for Adam optimizer

# Number of complete passes through the entire training dataset
NUM_EPOCHS = 10  # Moderate number suitable for initial training

# Target length for resampled EKG signals (standardizes input size)
TARGET_LENGTH = 1000  # Samples per lead after resampling

# ------------------------------------------------------------------------------
# Diagnostic Classification Targets
# ------------------------------------------------------------------------------

# Diagnostic classes the model will learn to identify
# Using standardized SCP-ECG statement codes:
DIAGNOSES = ["NORM", "MI", "STTC", "CD", "HYP"]
# NORM: Normal ECG
# MI: Myocardial infarction (heart attack)
# STTC: ST/T-changes (ischemia or other abnormalities)
# CD: Conduction disturbance (e.g., bundle branch blocks)
# HYP: Hypertrophy (chamber enlargement)

# ------------------------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------------------------

# Random seed for reproducibility across runs
SEED = 42  # Classic "answer to the ultimate question" seed value

# Proportion of data reserved for testing (holdout set)
TEST_SIZE = 0.1  # 10% of data for final evaluation

# Proportion of training data reserved for validation during training
VAL_SIZE = 0.1  # 10% of training data for epoch-wise validation

# ------------------------------------------------------------------------------
# Hardware Configuration
# ------------------------------------------------------------------------------

# Determine available computational hardware for PyTorch
# Prefer GPU acceleration if available, fall back to CPU otherwise
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Note: 'cuda' refers to NVIDIA GPUs with CUDA support