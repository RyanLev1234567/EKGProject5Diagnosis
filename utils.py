"""
Utility Functions Module for ECG Analysis Pipeline

This module provides common utility functions used throughout the ECG analysis pipeline,
including logging configuration, device setup, and data loading helpers.
"""

import logging
import torch
import os
import json
import config

def get_logger(name="ekg_pipeline"):
    """
    Configure and return a logger with standardized formatting.
    
    This function sets up a logger with consistent formatting across all modules
    in the pipeline, ensuring uniform log message appearance and easy debugging.
    
    Args:
        name (str, optional): Name for the logger instance. 
                             Defaults to "ekg_pipeline".
                             
    Returns:
        logging.Logger: Configured logger instance with stream handler
    """
    # Configure basic logging settings
    logging.basicConfig(
        level=logging.INFO,  # Set default logging level to INFO
        format="%(asctime)s - %(levelname)s - %(message)s",  # Standard log format
        handlers=[
            logging.StreamHandler()  # Output logs to console/stdout
        ]
    )
    return logging.getLogger(name)

def setup_device():
    """
    Configure and return the appropriate computational device.
    
    This function detects available hardware and sets the device configuration
    for PyTorch operations. It prioritizes GPU acceleration when available
    for faster computation.
    
    Returns:
        str: Device identifier ('cuda' for GPU, 'cpu' for CPU)
        
    Note:
        Also updates the global config.DEVICE variable for consistent
        device usage across the application.
    """
    # Check for CUDA-enabled GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Update global configuration with detected device
    config.DEVICE = device
    
    return device

def load_class_counts():
    """
    Load class distribution statistics from the prepared data directory.
    
    This function reads the precomputed class counts JSON file that contains
    information about the distribution of diagnostic classes in the dataset.
    This is useful for understanding dataset balance and for potential
    class-weighted training strategies.
    
    Returns:
        dict or None: Dictionary containing class count statistics if file exists,
                     None if the file is not found
                     
    Example returned dictionary structure:
        {
            "total_rows": 1000,
            "with_any_superclass": 800,
            "class_counts": {
                "NORM": 300,
                "MI": 200,
                "STTC": 150,
                "CD": 100,
                "HYP": 50
            }
        }
    """
    # Construct path to class counts JSON file
    counts_path = os.path.join(config.PREPARED_DIR, "class_counts.json")
    
    try:
        # Attempt to read and parse the JSON file
        with open(counts_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Handle missing file gracefully with warning log
        logger = get_logger()
        logger.warning("Class counts file not found at %s", counts_path)
        return None