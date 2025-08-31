"""
Signal Preprocessing Utilities for ECG Data

This module provides signal processing functions for preparing ECG data for machine learning.
It includes normalization, resampling, and a complete preprocessing pipeline to ensure
consistent input format for neural network models.
"""

import numpy as np
from scipy.signal import resample  # For signal resampling functionality
import config  # Project configuration

def normalize_signal(X):
    """
    Normalize each ECG lead to have zero mean and unit variance.
    
    This standardization process ensures that all leads contribute equally to the model
    regardless of their original amplitude ranges, which improves training stability
    and model performance.
    
    Args:
        X (numpy.ndarray): Input ECG signal with shape [samples, leads]
        
    Returns:
        numpy.ndarray: Normalized signal with same shape as input
        
    Note:
        Adds a small epsilon (1e-8) to the denominator to prevent division by zero
        in case of constant signals (which should not occur in valid ECG data).
    """
    # Subtract mean and divide by standard deviation for each lead independently
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

def resample_signal(X, target_len=config.TARGET_LENGTH):
    """
    Resample ECG signal to a fixed target length.
    
    Resampling ensures consistent input size for neural networks, which require
    fixed-length inputs. This is particularly important for batch processing.
    
    Args:
        X (numpy.ndarray): Input ECG signal with shape [samples, leads]
        target_len (int, optional): Target length for resampling. 
                                  Defaults to config.TARGET_LENGTH.
                                  
    Returns:
        numpy.ndarray: Resampled signal with shape [target_len, leads]
        
    Note:
        Uses scipy's resample function which employs Fourier method for
        high-quality signal resampling with minimal distortion.
    """
    return resample(X, target_len)

def preprocess_pipeline(X):
    """
    Complete preprocessing pipeline for ECG signals.
    
    Applies all necessary preprocessing steps in the correct order:
    1. Resampling to fixed length
    2. Normalization to zero mean and unit variance
    
    This pipeline ensures raw ECG signals are transformed into a format
    suitable for machine learning model consumption.
    
    Args:
        X (numpy.ndarray): Raw ECG signal with arbitrary length and shape [samples, leads]
        
    Returns:
        numpy.ndarray: Preprocessed signal ready for model input with 
                      shape [config.TARGET_LENGTH, leads]
    """
    # First resample to ensure consistent length
    X = resample_signal(X)
    # Then normalize to ensure consistent scale across leads
    X = normalize_signal(X)
    return X