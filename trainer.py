"""
Model Training Module for ECG Diagnosis

This module handles the complete training pipeline for the CNN-LSTM-Attention model
on the PTB-XL ECG dataset. It includes data loading, model initialization, training loop,
and saving of trained weights and training history.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_builder import build_dataset  # Custom dataset builder
from model import CNN_LSTM_Attention  # Neural network architecture
from tqdm import tqdm  # Progress bar utility
import numpy as np
from utils import get_logger  # Custom logging utility
import config  # Project configuration
import os
import json

# Initialize logger for this module
logger = get_logger()

# Import diagnostic classes from configuration
DIAGNOSES = config.DIAGNOSES

def train_model():
    """
    Complete model training pipeline.
    
    This function:
    1. Loads and prepares the training dataset
    2. Initializes the model and training components
    3. Executes the training loop for specified number of epochs
    4. Saves the trained model weights and training history
    5. Logs training progress and metrics
    
    Raises:
        RuntimeError: If no training data is loaded
    """
    
    # Path to the training dataset CSV file
    train_csv = os.path.join(config.PREPARED_DIR, "train.csv")
    
    logger.info("Building training dataset from: %s", train_csv)
    # Build the dataset using our custom dataset builder
    dataset = build_dataset(train_csv, config.RECORDS_DIR)
    
    # Validate that data was loaded successfully
    if len(dataset) == 0:
        error_msg = "No training data loaded! Please check file paths and ECG record locations."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Create DataLoader for batch processing during training
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True  # Shuffle for better training convergence
    )

    # Initialize the neural network model
    model = CNN_LSTM_Attention(
        input_channels=12,  # Standard 12-lead ECG input
        num_classes=len(DIAGNOSES)  # Number of diagnostic classes
    ).to(config.DEVICE)  # Move model to appropriate device (GPU/CPU)
    
    # Define loss function - Binary Cross Entropy with Logits for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Define optimizer - Adam with learning rate from configuration
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    logger.info("Starting training for %d epochs with %d samples...", 
                config.NUM_EPOCHS, len(dataset))
    
    # Dictionary to track training history for analysis and visualization
    training_history = {
        'epoch_losses': [],  # Average loss per epoch
        'batch_losses': []   # Loss for each individual batch
    }
    
    # Main training loop
    for epoch in range(config.NUM_EPOCHS):
        # Set model to training mode (enables dropout, batch norm, etc.)
        model.train()
        running_loss = 0.0  # Accumulated loss for the epoch
        epoch_batch_losses = []  # Store losses for each batch in this epoch

        # Create progress bar for batches within this epoch
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", leave=False)
        
        # Process each batch in the dataloader
        for batch_idx, (X_batch, y_batch) in enumerate(batch_pbar):
            # Move data to appropriate device (GPU/CPU)
            X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            
            # Reset gradients from previous batch
            optimizer.zero_grad()
            
            # Forward pass: compute model predictions
            outputs = model(X_batch)
            
            # Calculate loss between predictions and ground truth
            loss = criterion(outputs, y_batch)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            # Update running loss (weighted by batch size)
            running_loss += loss.item() * X_batch.size(0)
            current_loss = loss.item()
            epoch_batch_losses.append(current_loss)
            
            # Update progress bar with current loss metrics
            batch_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',  # Current batch loss
                'avg_loss': f'{np.mean(epoch_batch_losses):.4f}'  # Running average loss
            })

        # Calculate average loss for this epoch
        epoch_loss = running_loss / len(dataset)
        
        # Store epoch metrics in history
        training_history['epoch_losses'].append(epoch_loss)
        training_history['batch_losses'].extend(epoch_batch_losses)
        
        # Log epoch completion with loss metric
        logger.info("Epoch %d/%d completed - Average Loss: %.4f", 
                   epoch+1, config.NUM_EPOCHS, epoch_loss)

    # Save the trained model weights
    model_save_path = os.path.join(config.MODELS_DIR, "cnn_lstm_attention_5diagnoses.pth")
    torch.save(model.state_dict(), model_save_path)
    logger.info("Trained model saved to: %s", model_save_path)
    
    # Save training history for later analysis and visualization
    history_path = os.path.join(config.MODELS_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("Training completed successfully! Training history saved to: %s", history_path)

if __name__ == "__main__":
    """
    Main entry point when script is executed directly.
    Runs the complete training pipeline.
    """
    train_model()