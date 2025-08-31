"""
Configuration and Path Settings for EKG Analysis Project

This module centralizes all file paths, model parameters, and configuration settings
to ensure consistency across the entire EKG analysis pipeline. It serves as the
single source of truth for directory structures, hyperparameters, and hardware configuration.
"""

import os
import torch
import json

# ------------------------------------------------------------------------------
# Directory Path Configuration
# ------------------------------------------------------------------------------

# Root directory of the project - contains all data, code, and results
# Using absolute path to ensure consistency across different execution environments
BASE_DIR = r"C:\Users\15166\OneDrive\Desktop\final_ekg"

# Directory containing raw EKG record files (likely in WFDB format)
# The 'records100' suggests these may be 100Hz sampled records
RECORDS_DIR = os.path.join(BASE_DIR, "records100")

# Directory for preprocessed and prepared data ready for model training
# Contains normalized, resampled, and formatted EKG data
PREPARED_DIR = os.path.join(BASE_DIR, "prepared")

# Directory for saving trained model weights and architectures
MODELS_DIR = os.path.join(BASE_DIR, "models")
# Create models directory if it doesn't exist (prevents errors during model saving)
# exist_ok=True prevents errors if directory already exists
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Data File Paths
# ------------------------------------------------------------------------------

# PTB-XL database CSV file containing metadata and diagnostic information
# This is a comprehensive clinical ECG dataset with expert annotations
PTBXL_CSV = os.path.join(BASE_DIR, "ptbxl_database.csv")

# SCP statements CSV file containing standardized diagnostic codes and descriptions
# SCP (Standard Communication Protocol) provides consistent diagnostic terminology
SCP_CSV = os.path.join(BASE_DIR, "scp_statements.csv")

# ------------------------------------------------------------------------------
# Model Training Hyperparameters
# ------------------------------------------------------------------------------

# Number of samples processed before model parameters are updated
# Balance between memory usage (lower) and training stability (higher)
BATCH_SIZE = 32

# Step size for parameter updates during optimization
# 1e-3 is a common starting point for Adam optimizer in deep learning
LEARNING_RATE = 1e-3

# Number of complete passes through the entire training dataset
# Increased from 10 to 15 epochs for potentially better convergence
NUM_EPOCHS = 15

# Target length for resampled EKG signals (standardizes input size)
# 1000 samples provides good temporal resolution while managing computational cost
TARGET_LENGTH = 1000

# ------------------------------------------------------------------------------
# Diagnostic Classification Targets
# ------------------------------------------------------------------------------

# Diagnostic classes the model will learn to identify
# Using standardized SCP-ECG statement codes for clinical relevance:
DIAGNOSES = ["NORM", "MI", "STTC", "CD", "HYP"]
# NORM: Normal ECG rhythm
# MI: Myocardial infarction (heart attack)
# STTC: ST/T-wave changes (indicating ischemia or other abnormalities)
# CD: Conduction disturbance (e.g., bundle branch blocks, AV blocks)
# HYP: Hypertrophy (ventricular or atrial chamber enlargement)

# ------------------------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------------------------

# Random seed for reproducibility across runs
# Using 42 for consistency with common machine learning practices
SEED = 42

# Proportion of data reserved for testing (holdout set)
# 10% of data for final model evaluation on unseen examples
TEST_SIZE = 0.1

# Proportion of training data reserved for validation during training
# 10% of training data for epoch-wise performance monitoring
VAL_SIZE = 0.1

# ------------------------------------------------------------------------------
# Hardware Configuration
# ------------------------------------------------------------------------------

# Determine available computational hardware for PyTorch
# Prefer GPU acceleration if available for faster training, fall back to CPU otherwise
# 'cuda' refers to NVIDIA GPUs with CUDA support, which significantly speeds up training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
