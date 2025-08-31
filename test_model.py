"""
Model Evaluation Module for ECG Diagnosis

This module provides comprehensive evaluation of the trained ECG diagnosis model
on the test dataset. It calculates multiple performance metrics, generates detailed
reports, and handles the complete evaluation pipeline for multi-label classification.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import wfdb  # Waveform Database library for reading physiological signals
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from model import CNN_LSTM_Attention  # Our custom neural network architecture
from preprocess import normalize_signal, resample_signal  # Signal preprocessing functions
from tqdm import tqdm  # Progress bar utility
from utils import get_logger  # Custom logging utility
import config  # Project configuration

# Initialize logger for this module
logger = get_logger()

# Import diagnostic classes from configuration
DIAGNOSES = config.DIAGNOSES

class ECGTestDataset(Dataset):
    """
    PyTorch Dataset class for loading and preprocessing test ECG data.
    
    This specialized dataset handles loading of ECG records, preprocessing,
    and label preparation specifically for model evaluation.
    """
    
    def __init__(self, csv_path, records_path):
        """
        Initialize the test dataset.
        
        Args:
            csv_path (str): Path to the CSV file containing test set metadata
            records_path (str): Base path to the directory containing ECG record files
        """
        # Load the test dataset metadata
        self.df = pd.read_csv(csv_path)
        self.records_path = records_path
        
    def __len__(self):
        """Return the number of samples in the test dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (X, y, ecg_id) where:
                X (torch.Tensor): Preprocessed ECG signal
                y (torch.Tensor): Multi-label target tensor
                ecg_id (str): Original ECG record identifier
        """
        row = self.df.iloc[idx]
        record_path = row['record_path']
        
        # Load ECG signal using WFDB library
        record = wfdb.rdrecord(record_path)
        X = record.p_signal  # Extract physiological signal data
        
        # Preprocess the signal: resample then normalize
        X = normalize_signal(resample_signal(X, config.TARGET_LENGTH))
        # Convert to PyTorch tensor with appropriate data type
        X = torch.tensor(X, dtype=torch.float)
        
        # Extract and encode diagnostic labels
        labels = json.loads(row['labels_json'])
        # Create multi-hot encoding: 1 if diagnosis present, 0 otherwise
        y = [1 if diag in labels else 0 for diag in DIAGNOSES]
        y = torch.tensor(y, dtype=torch.float)
        
        return X, y, row['ecg_id']

def evaluate_performance(all_labels, all_preds, all_probs):
    """
    Calculate comprehensive performance metrics for multi-label classification.
    
    Args:
        all_labels (numpy.ndarray): Ground truth labels [samples, classes]
        all_preds (numpy.ndarray): Binary predictions [samples, classes]
        all_probs (numpy.ndarray): Prediction probabilities [samples, classes]
        
    Returns:
        dict: Comprehensive performance metrics for each class and overall
    """
    # Dictionary to store results for each diagnostic class
    results = {}
    
    # Calculate metrics for each diagnostic class individually
    for i, diagnosis in enumerate(DIAGNOSES):
        # Skip evaluation if no positive samples exist for this class
        if np.sum(all_labels[:, i]) == 0:
            logger.warning("No positive samples for %s in test set", diagnosis)
            results[diagnosis] = {
                'auc': float('nan'),  # Not applicable
                'precision': 0.0,
                'recall': 0.0,
                'f1-score': 0.0,
                'support': 0,  # Number of positive samples
                'confusion_matrix': np.array([[0, 0], [0, 0]])  # Empty confusion matrix
            }
            continue
            
        # Calculate AUC-ROC (Area Under the Receiver Operating Characteristic curve)
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            # Handle cases where AUC calculation fails (e.g., only one class present)
            auc = float('nan')
        
        # Calculate confusion matrix components manually
        tp = np.sum((all_preds[:, i] == 1) & (all_labels[:, i] == 1))  # True positives
        fp = np.sum((all_preds[:, i] == 1) & (all_labels[:, i] == 0))  # False positives
        fn = np.sum((all_preds[:, i] == 0) & (all_labels[:, i] == 1))  # False negatives
        tn = np.sum((all_preds[:, i] == 0) & (all_labels[:, i] == 0))  # True negatives
        
        # Calculate precision, recall, and F1-score with safety checks
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(np.sum(all_labels[:, i]))  # Number of positive samples
        
        # Create 2x2 confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Store results for this diagnostic class
        results[diagnosis] = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support,
            'confusion_matrix': cm
        }
    
    # Calculate micro-average AUC (considers all predictions equally)
    try:
        micro_auc = roc_auc_score(all_labels.ravel(), all_probs.ravel())
    except ValueError:
        micro_auc = float('nan')
    
    # Calculate overall accuracy
    accuracy = accuracy_score(all_labels.ravel(), all_preds.ravel())
    
    # Calculate macro averages (simple mean across classes)
    macro_precision = np.mean([results[d]['precision'] for d in DIAGNOSES if d in results])
    macro_recall = np.mean([results[d]['recall'] for d in DIAGNOSES if d in results])
    macro_f1 = np.mean([results[d]['f1-score'] for d in DIAGNOSES if d in results])
    
    # Calculate weighted averages (weighted by class support)
    weights = [results[d]['support'] for d in DIAGNOSES if d in results]
    weighted_precision = np.average([results[d]['precision'] for d in DIAGNOSES if d in results], weights=weights)
    weighted_recall = np.average([results[d]['recall'] for d in DIAGNOSES if d in results], weights=weights)
    weighted_f1 = np.average([results[d]['f1-score'] for d in DIAGNOSES if d in results], weights=weights)
    
    # Store overall metrics
    results['overall'] = {
        'micro_auc': micro_auc,
        'accuracy': accuracy,
        'macro_avg': {'precision': macro_precision, 'recall': macro_recall, 'f1-score': macro_f1},
        'weighted_avg': {'precision': weighted_precision, 'recall': weighted_recall, 'f1-score': weighted_f1}
    }
    
    return results

def print_detailed_report(results):
    """
    Print a comprehensive performance report to the console.
    
    Args:
        results (dict): Performance metrics dictionary from evaluate_performance
    """
    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    
    # Print per-class metrics
    print("\nPER-CLASS METRICS:")
    print("-" * 60)
    for diagnosis in DIAGNOSES:
        if diagnosis in results:
            metrics = results[diagnosis]
            print(f"{diagnosis:5s} | AUC: {metrics['auc']:.3f} | "
                  f"Precision: {metrics['precision']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"F1: {metrics['f1-score']:.3f} | "
                  f"Support: {metrics['support']}")
    
    # Print overall summary metrics
    overall = results['overall']
    print("\nOVERALL METRICS:")
    print("-" * 60)
    print(f"Micro AUC: {overall['micro_auc']:.3f}")
    print(f"Accuracy: {overall['accuracy']:.3f}")
    print(f"Macro Precision: {overall['macro_avg']['precision']:.3f}")
    print(f"Macro Recall: {overall['macro_avg']['recall']:.3f}")
    print(f"Macro F1: {overall['macro_avg']['f1-score']:.3f}")
    print(f"Weighted Precision: {overall['weighted_avg']['precision']:.3f}")
    print(f"Weighted Recall: {overall['weighted_avg']['recall']:.3f}")
    print(f"Weighted F1: {overall['weighted_avg']['f1-score']:.3f}")

def evaluate_model():
    """
    Complete model evaluation pipeline.
    
    This function:
    1. Loads the trained model
    2. Prepares the test dataset
    3. Runs inference on all test samples
    4. Calculates comprehensive performance metrics
    5. Prints detailed evaluation report
    
    Returns:
        dict: Comprehensive performance results
    """
    
    # Load the trained model weights
    model_path = os.path.join(config.MODELS_DIR, "cnn_lstm_attention_5diagnoses.pth")
    logger.info("Loading trained model from: %s", model_path)
    
    # Initialize model with correct architecture
    model = CNN_LSTM_Attention(input_channels=12, num_classes=len(DIAGNOSES))
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    # Move model to appropriate device and set to evaluation mode
    model.to(config.DEVICE)
    model.eval()  # Disables dropout and batch normalization
    
    # Create test dataset and dataloader
    test_csv_path = os.path.join(config.PREPARED_DIR, "test.csv")
    logger.info("Loading test data from: %s", test_csv_path)
    test_dataset = ECGTestDataset(test_csv_path, config.RECORDS_DIR)
    # Create data loader for batch processing
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Initialize lists to store evaluation results
    all_preds, all_probs, all_labels, all_ecg_ids = [], [], [], []
    
    logger.info("Running model evaluation on test set...")
    # Perform inference with gradient computation disabled
    with torch.no_grad():
        for X_batch, y_batch, ecg_ids in tqdm(test_loader, desc="Evaluating", total=len(test_loader)):
            # Move data to appropriate device (GPU/CPU)
            X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            
            # Forward pass through model
            outputs = model(X_batch)
            # Convert logits to probabilities using sigmoid activation
            probs = torch.sigmoid(outputs)
            # Apply threshold (0.5) to get binary predictions
            preds = (probs > 0.5).int()
            
            # Store results (move to CPU for numpy conversion)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_ecg_ids.extend(ecg_ids)
    
    # Concatenate all batch results into single arrays
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate comprehensive performance metrics
    results = evaluate_performance(all_labels, all_preds, all_probs)
    
    # Print detailed evaluation report to console
    print_detailed_report(results)
    
    logger.info("Model evaluation completed successfully!")
    return results

if __name__ == "__main__":
    """
    Main entry point when script is executed directly.
    Runs the complete evaluation pipeline.
    """
    evaluate_model()