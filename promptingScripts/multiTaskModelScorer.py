"""
Multi-Task Model Scorer
This script loads a model once and runs multiple cognitive tasks (DFE, BART, etc.) 
to score human decision-making behavior against the model's predictions.
"""

import os
import sys
import pandas as pd
import torch
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import gc

# Import task modules (assuming they're in the same directory)
try:
    from dfe_scorer import run_dfe_scoring, initialize_model_and_tokenizer as init_dfe_model
    from bart_scorer import run_bart_scoring, initialize_model_and_tokenizer as init_bart_model
except ImportError as e:
    print(f"Error importing task modules: {e}")
    print("Make sure dfe_scorer.py and bart_scorer.py are in the same directory")
    sys.exit(1)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Available models - add more as needed
AVAILABLE_MODELS = {
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
    # Add more models here
}

# Task configurations
TASK_CONFIGS = {
    "dfe": {
        "name": "DFE",
        "scorer_function": run_dfe_scoring,
        "file_suffix": "_dfe_data.json"
    },
    "bart": {
        "name": "BART", 
        "scorer_function": run_bart_scoring,
        "file_suffix": "_bart_data.json"
    },
    # Add more tasks here
}

class MultiTaskScorer:
    def __init__(self, model_name: str, output_dir: str = "outputs"):
        self.model_name = model_name
        self.model_key = model_name.split('/')[-1]
        self.output_dir = output_dir
        self.model_initialized = False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Track results
        self.all_results = []
        
    def initialize_model(self) -> bool:
        """Initialize the model once for all tasks."""
        logging.info(f"Initializing model: {self.model_name}")
        
        # Try initializing with DFE module first (both should work the same way)
        success = init_dfe_model(self.model_name)
        
        if success:
            self.model_initialized = True
            logging.info(f"Model {self.model_name} initialized successfully")
        else:
            logging.error(f"Failed to initialize model {self.model_name}")
            
        return success
    
    def run_task(self, task_name: str, input_file: str, test_mode: bool = False, 
                 verbose: bool = False) -> Optional[pd.DataFrame]:
        """Run a specific task."""
        if not self.model_initialized:
            logging.error("Model not initialized. Call initialize_model() first.")
            return None
            
        if task_name not in TASK_CONFIGS:
            logging.error(f"Unknown task: {task_name}. Available tasks: {list(TASK_CONFIGS.keys())}")
            return None
            
        if not os.path.exists(input_file):
            logging.error(f"Input file not found: {input_file}")
            return None
            
        task_config = TASK_CONFIGS[task_name]
        scorer_function = task_config["scorer_function"]
        
        logging.info(f"Running {task_config['name']} task on {input_file}")
        
        try:
            # Run the task - model is already loaded, so pass None for model_name
            results_df = scorer_function(
                input_file=input_file,
                model_name=None,  # Model already loaded
                model_key=self.model_key,
                test_mode=test_mode,
                verbose=verbose
            )
            
            if not results_df.empty:
                # Save task-specific results
                task_output_file = os.path.join(
                    self.output_dir, 
                    f"{self.model_key}_{task_name}_results.csv"
                )
                results_df.to_csv(task_output_file, index=False)
                logging.info(f"Saved {len(results_df)} {task_name.upper()} results to {task_output_file}")
                
                # Add to combined results
                self.all_results.append(results_df)
                
                return results_df
            else:
                logging.warning(f"No results from {task_name} task")
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Error running {task_name} task: {e}")
            return None
    
    def run_all_tasks(self, input_dir: str, test_mode: bool = False, 
                     verbose: bool = False, tasks: Optional[List[str]] = None) -> bool:
        """Run all specified tasks on data files in input directory."""
        if not self.model_initialized:
            if not self.initialize_model():
                return False
        
        tasks_to_run = tasks or list(TASK_CONFIGS.keys())
        
        logging.info(f"Running tasks: {tasks_to_run}")
        
        for task_name in tasks_to_run:
            if task_name not in TASK_CONFIGS:
                logging.warning(f"Skipping unknown task: {task_name}")
                continue
                
            # Look for input file
            task_config = TASK_CONFIGS[task_name]
            file_suffix = task_config["file_suffix"]
            
            # Try different file patterns
            possible_files = [
                os.path.join(input_dir, f"{task_name}{file_suffix}"),
                os.path.join(input_dir, f"{task_name}_data.json"),
                os.path.join(input_dir, f"{task_name}.json"),
                os.path.join(input_dir, f"{task_name}_data.jsonl"),
                os.path.join(input_dir, f"{task_name}.jsonl"),
            ]
            
            input_file = None
            for file_path in possible_files:
                if os.path.exists(file_path):
                    input_file = file_path
                    break
            
            if input_file:
                self.run_task(task_name, input_file, test_mode, verbose)
            else:
                logging.warning(f"No input file found for {task_name} task. Tried: {possible_files}")
        
        return True
    
    def save_combined_results(self) -> str:
        """Save all results to a single CSV file."""
        if not self.all_results:
            logging.warning("No results to save")
            return ""
            
        # Combine all results
        combined_df = pd.concat(self.all_results, ignore_index=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = os.path.join(
            self.output_dir, 
            f"{self.model_key}_all_tasks_{timestamp}.csv"
        )
        
        combined_df.to_csv(combined_file, index=False)
        logging.info(f"Saved combined results ({len(combined_df)} rows) to {combined_file}")
        
        # Print summary
        print(f"\n--- COMBINED RESULTS SUMMARY ---")
        print(f"Model: {self.model_name}")
        print(f"Total decisions scored: {len(combined_df)}")
        print(f"Tasks: {combined_df['task'].unique()}")
        print(f"Participants: {combined_df['part_id'].nunique()}")
        
        if len(combined_df) > 0:
            print("\nDecisions per task:")
            print(combined_df['task'].value_counts())
            
            print("\nParticipants per task:")
            print(combined_df.groupby('task')['part_id'].nunique())
        
        return combined_file
    
    def cleanup(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logging.info("Memory cleanup completed")

def main():
    parser = argparse.ArgumentParser(description="Score multiple cognitive tasks with language models")
    parser.add_argument("--model", type=str, required=True,
                        help=f"Model to use. Options: {list(AVAILABLE_MODELS.keys())} or full model path")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing input data files")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save results (default: outputs)")
    parser.add_argument("--tasks", nargs="+", 
                        help=f"Tasks to run. Options: {list(TASK_CONFIGS.keys())} (default: all)")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode (limited participants)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--no-combined", action="store_true",
                        help="Skip saving combined results file")
    
    args = parser.parse_args()
    
    # Resolve model name
    if args.model in AVAILABLE_MODELS:
        model_name = AVAILABLE_MODELS[args.model]
    else:
        model_name = args.model  # Assume it's a full path
    
    if not os.path.exists(args.input_dir):
        logging.error(f"Input directory not found: {args.input_dir}")
        return
    
    # Initialize scorer
    scorer = MultiTaskScorer(model_name, args.output_dir)
    
    try:
        # Run tasks
        success = scorer.run_all_tasks(
            input_dir=args.input_dir,
            test_mode=args.test,
            verbose=args.verbose,
            tasks=args.tasks
        )
        
        if success and not args.no_combined:
            scorer.save_combined_results()
            
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Error during execution: {e}")
    finally:
        # Cleanup
        scorer.cleanup()

def batch_run_models(models: List[str], input_dir: str, output_dir: str = "outputs", 
                    tasks: Optional[List[str]] = None, test_mode: bool = False) -> Dict[str, str]:
    """
    Convenience function to run multiple models sequentially.
    Returns dictionary mapping model names to their combined results files.
    """
    results_files = {}
    
    for model_key in models:
        if model_key in AVAILABLE_MODELS:
            model_name = AVAILABLE_MODELS[model_key]
        else:
            model_name = model_key
            
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        scorer = MultiTaskScorer(model_name, output_dir)
        
        try:
            success = scorer.run_all_tasks(input_dir, test_mode=test_mode, tasks=tasks)
            if success:
                combined_file = scorer.save_combined_results()
                results_files[model_key] = combined_file
        except Exception as e:
            logging.error(f"Error processing model {model_name}: {e}")
        finally:
            scorer.cleanup()
    
    return results_files

if __name__ == "__main__":
    main()