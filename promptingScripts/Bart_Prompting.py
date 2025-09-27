#!/usr/bin/env python3
"""
BART Task Module - Standalone task for BART evaluation.
This module can be imported and run by the model manager.
"""

import json
import torch
import pandas as pd
import re
import logging
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# Task-specific configuration
DATA_FILE = "bart_data/prompts_bart.jsonl"  

def run_task(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_key: str, 
             test_mode: bool = False, data_file: str = DATA_FILE) -> pd.DataFrame:
    """
    Main task runner function. This is the interface that the model manager will call.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        model_key: String identifier for the model
        test_mode: Whether to run in test mode (fewer entries)
        data_file: Path to the data file
    
    Returns:
        pandas.DataFrame: Results dataframe
    """
    logging.info(f"Starting TEMPLATE task for model: {model_key}")
    
    all_rows = []
    
    try:
        # Load your data here
        # Example for JSONL format:
        # with open(data_file) as f:
        #     entries = [json.loads(line) for line in f]
        
        # Or for CSV:
        # df = pd.read_csv(data_file)
        # entries = df.to_dict('records')
        
        # Placeholder data loading
        entries = [{"text": "example", "label": "test"}]
        
        # Limit entries in test mode
        if test_mode:
            entries = entries[:2]
            logging.info(f"Test mode: processing only first 2 entries")
            
        # Process each entry
        for entry_idx, entry in enumerate(entries):
            logging.info(f"Processing TEMPLATE entry {entry_idx + 1}/{len(entries)}")
            
            try:
                # TODO: Implement your task-specific processing here
                # This is where you would:
                # 1. Format the input for your model
                # 2. Run inference
                # 3. Extract the results you need
                
                # Example row structure - modify as needed
                result_row = {
                    "model": model_key,
                    "entry_id": entry_idx,
                    "input_text": entry.get("text", ""),
                    "result": "placeholder_result",  # Replace with actual result
                    # Add more columns as needed for your task
                }
                all_rows.append(result_row)
                    
            except Exception as e:
                logging.error(f"Error processing TEMPLATE entry {entry_idx}: {e}")
                continue
        
        # Create and return DataFrame
        if all_rows:
            df = pd.DataFrame(all_rows)
            logging.info(f"TEMPLATE task completed. Generated {len(all_rows)} rows of results.")
            return df
        else:
            logging.warning(f"No results generated for TEMPLATE task on {model_key}")
            return pd.DataFrame()  # Return empty DataFrame
            
    except Exception as e:
        logging.error(f"Error in TEMPLATE task for model {model_key}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def get_task_info() -> Dict[str, str]:
    """Return information about this task."""
    return {
        "name": "TEMPLATE",  # Change this
        "description": "Template task description",  # Change this
        "output_columns": ["model", "entry_id", "input_text", "result"],  # Update this
        "data_file": DATA_FILE
    }

# Helper functions for your specific task
def your_helper_function():
    """Add any helper functions you need here."""
    pass

# For standalone testing
if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
    print("Task info:", get_task_info())