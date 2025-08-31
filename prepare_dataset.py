"""
PTB-XL Data Preparation Module

This module prepares the PTB-XL ECG dataset for machine learning by:
1. Loading and processing metadata
2. Mapping SCP codes to diagnostic superclasses
3. Creating multi-label annotations
4. Generating train/validation/test splits with stratification
5. Saving prepared datasets with updated file paths
"""

import os
import json
import ast  # For safely evaluating string representations of Python literals
import pandas as pd
from sklearn.model_selection import train_test_split  # For creating data splits
from tqdm import tqdm  # For progress bars during processing
from utils import get_logger  # Custom logging utility
import config  # Project configuration

# Initialize logger for this module
logger = get_logger()

# Import diagnostic superclasses from configuration
SUPERCLASSES = config.DIAGNOSES

def prepare_data_splits():
    """
    Prepare train/validation/test splits from PTB-XL dataset with updated file paths.
    
    This function performs the complete data preparation pipeline including:
    - Loading and parsing metadata
    - Mapping SCP codes to diagnostic classes
    - Creating multi-label annotations
    - Generating stratified splits
    - Saving prepared datasets to disk
    """
    
    logger.info("Loading PTB-XL metadata from: %s", config.PTBXL_CSV)
    # Load the main PTB-XL database CSV file
    df = pd.read_csv(config.PTBXL_CSV)
    # Convert string representation of SCP codes to Python dictionary
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    logger.info("Loading SCP statement mapping from: %s", config.SCP_CSV)
    # Load SCP statements CSV which contains diagnostic code mappings
    scp_df = pd.read_csv(config.SCP_CSV, index_col=0)
    # Filter to only include diagnostic statements (exclude technical statements)
    if "diagnostic" in scp_df.columns:
        scp_df = scp_df[scp_df["diagnostic"] == 1]
    # Create mapping from SCP code to diagnostic superclass
    code_to_super = scp_df["diagnostic_class"].to_dict()

    logger.info("Building dataset rows with labels and record paths...")
    # Initialize data structures for tracking
    rows = []  # Will store processed row data
    class_counts = {c: 0 for c in SUPERCLASSES}  # Count occurrences of each diagnostic class
    with_any = 0  # Count records with at least one diagnostic superclass
    
    def record_base_path(filename_lr):
        """
        Build the full file path for WFDB record reading.
        
        Args:
            filename_lr (str): Low-resolution filename from PTB-XL metadata
            
        Returns:
            str: Normalized full path to the record file
        """
        # Normalize path separators and handle relative paths
        rel = filename_lr.replace("\\", "/")
        # Remove 'records100/' prefix if present
        if rel.startswith("records100/"):
            rel = rel[len("records100/"):]
        # Remove .dat extension for WFDB compatibility
        if rel.lower().endswith(".dat"):
            rel = rel[:-4]
        # Construct full normalized path
        return os.path.normpath(os.path.join(config.RECORDS_DIR, rel))
    
    def labels_from_scp_codes(scp_codes):
        """
        Extract diagnostic superclasses from SCP codes dictionary.
        
        Args:
            scp_codes (dict): Dictionary of SCP codes and their confidence values
            
        Returns:
            list: Sorted list of diagnostic superclasses present in the record
        """
        present = set()
        for code in scp_codes.keys():
            cls = code_to_super.get(code)
            # Only include classes that are in our target superclasses
            if cls in SUPERCLASSES:
                present.add(cls)
        return sorted(present)
    
    def one_hot(classes_present):
        """
        Create one-hot encoded representation of diagnostic classes.
        
        Args:
            classes_present (list): List of diagnostic classes present in the record
            
        Returns:
            dict: Dictionary with one-hot encoded labels for each superclass
        """
        return {f"label_{c}": (1 if c in classes_present else 0) for c in SUPERCLASSES}
    
    def pick_primary_class(classes_present):
        """
        Select a single primary class for stratification purposes.
        
        For multi-label records, prioritizes non-NORM classes to ensure
        balanced stratification across pathological conditions.
        
        Args:
            classes_present (list): List of diagnostic classes present
            
        Returns:
            str: Selected primary class for stratification
        """
        if not classes_present:
            return "NONE"  # No diagnostic classes present
        if len(classes_present) == 1:
            return classes_present[0]  # Single class case
        
        # For multiple classes, prefer non-NORM classes for stratification
        non_norm = [c for c in classes_present if c != "NORM"]
        return non_norm[0] if non_norm else classes_present[0]

    # Process each record in the dataset with progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing ECG records"):
        # Extract diagnostic labels from SCP codes
        labels = labels_from_scp_codes(row["scp_codes"])
        
        # Update statistics
        if labels:
            with_any += 1
            for c in labels:
                class_counts[c] += 1

        # Build full record path
        base_path = record_base_path(row["filename_lr"])
        # Create one-hot encoded labels
        oh = one_hot(labels)
        
        # Create processed row with all necessary information
        rows.append({
            "ecg_id": int(row["ecg_id"]),  # Original ECG identifier
            "record_path": base_path,  # Full path to record file
            "labels_json": json.dumps(labels),  # JSON string of labels for easy loading
            **oh,  # Unpack one-hot encoded labels
            "primary_class": pick_primary_class(labels)  # For stratification
        })

    # Create DataFrame from processed rows
    prepared = pd.DataFrame(rows)
    # Filter out records with no diagnostic classes
    prepared = prepared[prepared["primary_class"] != "NONE"].reset_index(drop=True)

    # Log dataset statistics
    total_rows = len(df)
    logger.info("Total rows in original CSV: %d", total_rows)
    logger.info("Rows with at least one diagnostic superclass: %d", with_any)
    logger.info("Class counts (multi-label presence):")
    for c in SUPERCLASSES:
        logger.info("  %s: %d", c, class_counts[c])
    
    # Save class distribution statistics to JSON file
    with open(os.path.join(config.PREPARED_DIR, "class_counts.json"), "w", encoding="utf-8") as f:
        json.dump({
            "total_rows": total_rows,
            "with_any_superclass": with_any,
            "class_counts": class_counts
        }, f, indent=2)

    # Get split sizes from configuration
    test_size = config.TEST_SIZE
    val_size = config.VAL_SIZE
    
    logger.info("Creating stratified train/validation/test splits...")
    # First split: separate training from temporary set (validation + test)
    train_df, temp_df = train_test_split(
        prepared,
        test_size=(test_size + val_size),  # Combined size for val + test
        random_state=config.SEED,  # For reproducibility
        stratify=prepared["primary_class"]  # Maintain class distribution
    )
    
    # Second split: separate validation from test
    # Calculate relative size of validation within the temp set
    relative_val = val_size / (test_size + val_size) if (test_size + val_size) > 0 else 0.5
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val),  # Remaining portion goes to test
        random_state=config.SEED,  # For reproducibility
        stratify=temp_df["primary_class"]  # Maintain class distribution
    )

    # Define columns to save in final CSV files
    cols = ["ecg_id", "record_path", "labels_json"] + [f"label_{c}" for c in SUPERCLASSES] + ["primary_class"]
    
    logger.info("Saving data splits to: %s", config.PREPARED_DIR)
    # Save each split to CSV files
    train_df[cols].to_csv(os.path.join(config.PREPARED_DIR, "train.csv"), index=False)
    val_df[cols].to_csv(os.path.join(config.PREPARED_DIR, "val.csv"), index=False)
    test_df[cols].to_csv(os.path.join(config.PREPARED_DIR, "test.csv"), index=False)

    # Final summary log
    logger.info("Data preparation completed successfully!")
    logger.info("Train set: %d rows", len(train_df))
    logger.info("Validation set: %d rows", len(val_df))
    logger.info("Test set: %d rows", len(test_df))

if __name__ == "__main__":
    """
    Main entry point when script is executed directly.
    Runs the complete data preparation pipeline.
    """
    prepare_data_splits()