"""
Main Pipeline Orchestrator - Coordinates Independent Execution of Pipeline Components

This module serves as the main entry point for the ECG analysis pipeline, providing
a command-line interface to run data preparation, model training, and evaluation
components either independently or as a complete workflow.
"""

import argparse  # For parsing command-line arguments
import time      # For tracking execution time and performance monitoring
from prepare_dataset import prepare_data_splits  # Data preparation component
from trainer import train_model                  # Model training component
from test_model import evaluate_model            # Model evaluation component
from utils import get_logger, setup_device       # Utility functions
import config                                    # Project configuration

# Initialize logging for this module
logger = get_logger()

def run_pipeline(args):
    """
    Execute the complete ECG analysis pipeline based on provided arguments.
    
    This function orchestrates the execution of pipeline components in sequence,
    with timing and logging for monitoring performance and progress.
    
    Args:
        args (argparse.Namespace): Command-line arguments specifying which 
                                  pipeline components to execute
    """
    # Configure hardware settings (GPU/CPU) for optimal performance
    setup_device()
    logger.info("Using computational device: %s", config.DEVICE)
    
    # Start timing the entire pipeline execution
    start_time = time.time()
    
    # Data Preparation Phase
    if args.prepare_data:
        logger.info("Initiating data preparation and splitting...")
        prepare_data_splits()  # Preprocess data and create train/val/test splits
    
    # Model Training Phase
    if args.train_model:
        logger.info("Starting model training process...")
        train_model()  # Train the neural network model on prepared data
    
    # Model Evaluation Phase
    if args.evaluate_model:
        logger.info("Commencing model evaluation...")
        evaluate_model()  # Assess model performance on test data
    
    # Calculate and log total execution time
    total_time = time.time() - start_time
    logger.info("Pipeline execution completed in %.2f seconds", total_time)
    
    # Final success message if complete pipeline was run
    if args.all:
        logger.info("Complete ECG analysis pipeline finished successfully!")

if __name__ == "__main__":
    """
    Main entry point when script is executed directly.
    
    Sets up command-line argument parsing and initiates pipeline execution.
    """
    # Configure command-line argument parser with descriptive help text
    parser = argparse.ArgumentParser(
        description="ECG Analysis Pipeline - Orchestrates data preparation, model training, and evaluation"
    )
    
    # Define individual pipeline component flags
    parser.add_argument(
        "--prepare_data", 
        action="store_true", 
        help="Prepare and split dataset into training, validation, and test sets"
    )
    parser.add_argument(
        "--train_model", 
        action="store_true", 
        help="Train the neural network model on the prepared dataset"
    )
    parser.add_argument(
        "--evaluate_model", 
        action="store_true", 
        help="Evaluate model performance on the test dataset"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run the complete pipeline end-to-end (equivalent to all individual flags)"
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # If --all flag is provided, enable all pipeline components
    if args.all:
        args.prepare_data = True
        args.train_model = True
        args.evaluate_model = True
    
    # Validate that at least one pipeline component was requested
    if not any([args.prepare_data, args.train_model, args.evaluate_model]):
        # Display help message if no valid arguments provided
        parser.print_help()
    else:
        # Execute the pipeline with the requested components
        run_pipeline(args)