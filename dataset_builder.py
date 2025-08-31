"""
ECG Dataset Module for PyTorch

This module provides a custom PyTorch Dataset class and data loading utilities
for handling electrocardiogram (ECG) data, including signal preprocessing and
label encoding for multi-label classification tasks.
"""

import torch
from torch.utils.data import Dataset  # Base class for custom datasets in PyTorch
import pandas as pd
import json
import wfdb  # Library for reading waveform database files
import numpy as np
from preprocess import normalize_signal, resample_signal  # Custom signal processing functions
from tqdm import tqdm  # Progress bar utility
from utils import get_logger  # Custom logging utility
import config  # Project configuration

# Initialize logger for this module
logger = get_logger()

class ECGDataset(Dataset):
    """
    A custom PyTorch Dataset class for ECG signal data.
    
    This class handles the storage and retrieval of preprocessed ECG signals
    and their corresponding multi-label diagnostic annotations.
    """
    
    def __init__(self, X_list, y_list):
        """
        Initialize the ECG dataset.
        
        Args:
            X_list (list): List of preprocessed ECG signal tensors
            y_list (list): List of multi-label target tensors
        """
        self.X_list = X_list  # Store ECG signals
        self.y_list = y_list  # Store diagnostic labels

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of ECG records in the dataset
        """
        return len(self.X_list)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (X, y) where:
                X (torch.Tensor): Preprocessed ECG signal tensor
                y (torch.Tensor): Multi-label diagnostic tensor
        """
        return self.X_list[idx], self.y_list[idx]

def build_dataset(csv_path, records_path):
    """
    Build an ECG dataset from prepared CSV files - Simplified version.
    
    This function loads ECG records, applies preprocessing, and converts
    diagnostic labels into a multi-hot encoded format suitable for training.
    
    Args:
        csv_path (str): Path to the CSV file containing dataset metadata
        records_path (str): Base path to the directory containing ECG record files
        
    Returns:
        ECGDataset: A PyTorch Dataset object containing preprocessed ECG signals
                   and their corresponding multi-label targets
    """
    logger.info("Loading dataset from prepared CSV: %s", csv_path)
    # Read the metadata CSV file containing record paths and labels
    df = pd.read_csv(csv_path)
    
    # Initialize lists to store processed data
    X_list, y_list = [], []
    error_count = 0
    success_count = 0
    
    logger.info("Processing %d ECG records from dataset...", len(df))
    
    # Process each ECG record with a progress bar for visual feedback
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading ECG signals"):
        try:
            # Get the record path from the prepared CSV
            record_path = row['record_path']
            
            # Load the ECG signal using WFDB library
            record = wfdb.rdrecord(record_path)
            # Extract the physiological signal data (typically voltage measurements)
            X = record.p_signal
            
            # Preprocess the signal: resample to consistent length
            X = resample_signal(X, config.TARGET_LENGTH)
            # Preprocess the signal: normalize to standard range
            X = normalize_signal(X)
            # Convert to PyTorch tensor with appropriate data type
            X_tensor = torch.tensor(X, dtype=torch.float)
            
            # Extract and process diagnostic labels
            labels_str = row['labels_json']
            # Parse JSON string to Python dictionary/list
            labels = json.loads(labels_str)
            # Convert to multi-hot encoding: 1 if diagnosis present, 0 otherwise
            y = [1 if diag in labels else 0 for diag in config.DIAGNOSES]
            # Convert to PyTorch tensor
            y_tensor = torch.tensor(y, dtype=torch.float)
            
            # Add successful processed data to lists
            X_list.append(X_tensor)
            y_list.append(y_tensor)
            success_count += 1
            
        except Exception as e:
            # Handle errors gracefully with logging and continue processing
            error_count += 1
            # Only show first few errors to avoid log spam
            if error_count <= 3:
                logger.warning("Error loading record %d: %s", row.get('ecg_id', idx), str(e))
            continue
    
    # Log final processing statistics
    logger.info("Successfully loaded %d ECG records", success_count)
    if error_count > 0:
        logger.warning("Failed to load %d records (see warnings above for details)", error_count)
    
    # Return the constructed dataset object
    return ECGDataset(X_list, y_list)