"""
Neural Network Architecture for ECG Signal Classification

This module defines a hybrid CNN-LSTM model with attention mechanism for
multi-class classification of electrocardiogram (ECG) signals. The architecture
combines convolutional layers for local feature extraction, LSTM layers for
temporal pattern recognition, and attention mechanisms for focusing on
clinically relevant segments of the ECG signal.
"""

import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    """
    Temporal Attention Mechanism for weighting important time steps.
    
    This module learns to assign importance weights to different time steps
    in the LSTM output sequence, allowing the model to focus on clinically
    relevant segments of the ECG signal.
    """
    
    def __init__(self, hidden_dim):
        """
        Initialize the temporal attention mechanism.
        
        Args:
            hidden_dim (int): Dimensionality of the hidden states from the LSTM
        """
        super(TemporalAttention, self).__init__()
        # Linear layer that computes attention scores for each time step
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        """
        Forward pass for temporal attention.
        
        Args:
            lstm_output (torch.Tensor): Output from LSTM layer with shape 
                                      [batch_size, seq_len, hidden_dim]
                                      
        Returns:
            tuple: (context, weights) where:
                context (torch.Tensor): Context vector representing the weighted 
                                      combination of all time steps [batch_size, hidden_dim]
                weights (torch.Tensor): Attention weights for each time step 
                                      [batch_size, seq_len, 1]
        """
        # lstm_output shape: [batch_size, sequence_length, hidden_dim]
        
        # Compute raw attention scores for each time step
        scores = self.attention(lstm_output)  # Shape: [batch_size, seq_len, 1]
        
        # Convert scores to probabilities using softmax (normalized over time dimension)
        weights = torch.softmax(scores, dim=1)  # Shape: [batch_size, seq_len, 1]
        
        # Compute context vector as weighted sum of all time steps
        context = torch.sum(weights * lstm_output, dim=1)  # Shape: [batch_size, hidden_dim]
        
        return context, weights


class CNN_LSTM_Attention(nn.Module):
    """
    Hybrid CNN-LSTM model with attention mechanism for ECG classification.
    
    Architecture overview:
    1. 1D CNN layers for local feature extraction from ECG leads
    2. Bidirectional LSTM for capturing temporal dependencies in both directions
    3. Temporal attention mechanism to focus on clinically relevant segments
    4. Fully connected layer for final classification
    
    This architecture is particularly well-suited for ECG analysis as it can
    capture both spatial relationships between leads and temporal patterns
    across the cardiac cycle.
    """
    
    def __init__(self, input_channels=12, num_classes=5, lstm_hidden=128):
        """
        Initialize the CNN-LSTM-Attention model.
        
        Args:
            input_channels (int): Number of input ECG leads (default: 12 for standard 12-lead ECG)
            num_classes (int): Number of diagnostic classes to predict (default: 5)
            lstm_hidden (int): Size of the LSTM hidden state (default: 128)
        """
        super(CNN_LSTM_Attention, self).__init__()
        
        # CNN feature extraction layers
        # Process each lead independently to extract local temporal features
        self.cnn = nn.Sequential(
            # First convolutional layer: extract basic waveform features
            # Input: [batch, 12 leads, seq_len] -> Output: [batch, 32 features, seq_len]
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),  # Non-linear activation for feature learning
            
            # Second convolutional layer: extract higher-level features
            # Input: [batch, 32 features, seq_len] -> Output: [batch, 64 features, seq_len]
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        # Bidirectional LSTM for temporal pattern recognition
        # Processes the sequence in both forward and backward directions
        # Input: [batch, seq_len, 64 features] -> Output: [batch, seq_len, lstm_hidden*2]
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=lstm_hidden, 
            batch_first=True,  # Input format: (batch, seq, feature)
            bidirectional=True  # Use both forward and backward passes
        )
        
        # Temporal attention mechanism
        # Input dimension is lstm_hidden*2 due to bidirectional LSTM
        self.attention = TemporalAttention(lstm_hidden * 2)
        
        # Final classification layer
        # Maps the attention-weighted context vector to class probabilities
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        """
        Forward pass through the complete network.
        
        Args:
            x (torch.Tensor): Input ECG tensor with shape [batch_size, seq_len, input_channels]
            
        Returns:
            torch.Tensor: Model predictions with shape [batch_size, num_classes]
        """
        # Input shape: [batch_size, sequence_length, 12_leads]
        
        # Permute dimensions for CNN: Conv1d expects [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # New shape: [batch_size, 12_leads, sequence_length]
        
        # Extract features using CNN layers
        x = self.cnn(x)  # Output shape: [batch_size, 64_features, sequence_length]
        
        # Permute back for LSTM: LSTM expects [batch, seq_len, features]
        x = x.permute(0, 2, 1)  # New shape: [batch_size, sequence_length, 64_features]
        
        # Process temporal patterns with bidirectional LSTM
        lstm_out, _ = self.lstm(x)  # Output shape: [batch_size, seq_len, lstm_hidden*2]
        
        # Apply temporal attention to focus on important time steps
        context, attn_weights = self.attention(lstm_out)  # Context shape: [batch_size, lstm_hidden*2]
        
        # Final classification
        out = self.fc(context)  # Output shape: [batch_size, num_classes]
        
        return out