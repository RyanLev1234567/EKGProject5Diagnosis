"""
PTB-XL Dataset Loader Module

This module provides a specialized loader for the PTB-XL electrocardiogram (ECG) dataset,
handling both the original dataset format and prepared data formats with fallback mechanisms.
It facilitates loading of ECG signals and their corresponding diagnostic labels.
"""

import os
import wfdb  # Waveform Database library for reading physiological signals
import ast   # Abstract Syntax Trees for safely evaluating string expressions
import pandas as pd
from utils import get_logger  # Custom logging utility
import config  # Project configuration module

# Initialize logger for this module
logger = get_logger()

class PTBXLLoader:
    """
    A specialized loader for the PTB-XL ECG dataset that handles multiple data formats.
    
    This class provides methods to load ECG records and their corresponding diagnostic
    labels from both the original PTB-XL format and prepared dataset formats.
    """
    
    def __init__(self, csv_path, base_path=config.RECORDS_DIR):
        """
        Initialize the PTB-XL dataset loader.
        
        Args:
            csv_path (str): Path to the CSV file containing dataset metadata
            base_path (str, optional): Base directory for record files. 
                                     Defaults to config.RECORDS_DIR.
        """
        # Load the metadata dataframe containing record information
        self.df = pd.read_csv(csv_path)
        # Store the base path for record files
        self.base_path = base_path

    def load_record(self, row, use_high_res=False):
        """
        Load a single ECG record and its corresponding labels.
        
        Args:
            row (pd.Series): A row from the metadata dataframe
            use_high_res (bool, optional): Whether to use high-resolution files. 
                                         Defaults to False (low-resolution).
                                         
        Returns:
            tuple: (X, y) where:
                X (numpy.ndarray): ECG signal data with shape (samples, leads)
                y (dict or list): Diagnostic labels in dictionary or list format
                
        Note: Handles both original PTB-XL format and prepared dataset formats
        with automatic fallback mechanisms.
        """
        # Determine the record path based on available columns in the dataframe
        if 'record_path' in row:
            # Use prepared dataset format with precomputed paths
            record_path = row['record_path']
        else:
            # Fallback to original PTB-XL dataset format
            # Select appropriate filename based on resolution preference
            fname = row['filename_hr'] if use_high_res else row['filename_lr']
            
            # Clean the filename path by removing any leading directory prefixes
            if fname.startswith('records100/'):
                fname = fname[len('records100/'):]
            
            # Construct full path and remove .dat extension for WFDB compatibility
            record_path = os.path.join(self.base_path, fname).replace('.dat','')
        
        # Read the ECG record using WFDB library
        # This loads the physiological signal data including voltage measurements
        record = wfdb.rdrecord(record_path)
        # Extract the physiological signal data (typically shape: [samples, leads])
        X = record.p_signal
        
        # Extract diagnostic labels based on available columns
        if 'labels_json' in row:
            # For prepared datasets with pre-processed label formats
            # Safely evaluate the string representation of labels
            y = ast.literal_eval(row['labels_json'])
        else:
            # Fallback to original PTB-XL format with SCP codes
            # SCP (Standard Communication Protocol) codes are standard ECG diagnoses
            y = ast.literal_eval(row['scp_codes'])
            
        return X, y

    def get_dataframe(self):
        """
        Get the underlying metadata dataframe.
        
        Returns:
            pd.DataFrame: The complete metadata dataframe containing information
                         about all records in the dataset, including file paths
                         and diagnostic labels.
        """
        return self.df