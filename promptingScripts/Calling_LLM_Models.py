#!/usr/bin/env python3
"""
Model Manager - Main orchestrator for running multiple tasks across multiple models.
Loads models once and runs multiple tasks on each model.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import argparse
import logging
import os
from typing import Dict, List, Any, Optional
import importlib.util
import sys

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Configuration Dictionary (Fine-tuned/Instruction Models Only) ---
MODEL_CONFIGS = {
    # --- HF Models -----
    "smollm2_1.7b": {
        "model_key": "SmolLM2-1.7B-Instruct",
        "model_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "is_test": False,
    },
    "SmolLM-1.7b": {
        "model_key": "SmolLM-1.7B-Instruct",
        "model_name": "HuggingFaceTB/SmolLM-1.7B-Instruct",
        "is_test": False,
    },
    
    "zephyr-7b": {
        "model_key": "zephyr-7b-beta",
        "model_name": "HuggingFaceH4/zephyr-7b-beta",
        "is_test": False,
    },
    
    # --- OLMo Instruct ---   
    "olmo2_7b_it": {
        "model_key": "OLMo-2-7B-Instruct",
        "model_name": "allenai/OLMo-2-1124-7B-Instruct",
        "is_test": False,
    },

    # ---- Llama Instruct Models ----
    "llama31_8b": {
        "model_key": "Llama-3.1-8B-Instruct",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "is_test": False,
    },
    "llama32_1b": {
        "model_key": "Llama-3.2-1B-Instruct",
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "is_test": False,
    },
    "llama32_3b": {
        "model_key": "Llama-3.2-3B-Instruct",
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "is_test": False,
    },
    # "llama31_70b_it": {
    #     "model_key": "Llama-3.1-70B-Instruct",
    #     "model_name": "meta-llama/Llama-3.1-70B-Instruct",
    #     "is_test": False,
    # },   
    # "llama33_70b_it": {
    #     "model_key": "Llama-3.3-70B-Instruct",
    #     "model_name": "meta-llama/Llama-3.3-70B-Instruct",
    #     "is_test": False,
    # }, 

    # --- Falcon Instruct ---
    "falcon3_1b": {
        "model_key": "Falcon-3-1B-Instruct",
        "model_name": "tiiuae/Falcon3-1B-Instruct",
        "is_test": False,
    },

    "falcon3_7b": {
        "model_key": "Falcon-3-7B-Instruct",
        "model_name": "tiiuae/Falcon3-7B-Instruct",
        "is_test": False,
    },

    "falcon3_10b": {
        "model_key": "Falcon-3-10B-Instruct",
        "model_name": "tiiuae/Falcon3-10B-Instruct",
        "is_test": False,
    },

    # --- Gemma Instruct ------
    "gemma-3-1b": {
        "model_key": "gemma-3-1b-it",
        "model_name": "google/gemma-3-1b-it",
        "is_test": False,
    },

     "gemma-3-4b": {
        "model_key": "gemma-3-4b-it",
        "model_name": "google/gemma-3-4b-it",
        "is_test": False,
    },

    # "gemma-3-12b": {
    #     "model_key": "gemma-3-12b-it",
    #     "model_name": "google/gemma-3-12b-it",
    #     "is_test": False,
    # },
    # "gemma-3-27b": {
    #     "model_key": "gemma-3-27b-it",
    #     "model_name": "google/gemma-3-27b-it",
    #     "is_test": False,
    # },

    "gemma-2-2b": {
        "model_key": "gemma-2-2b-it",
        "model_name": "google/gemma-2-2b-it",
        "is_test": False,
    },

    "gemma-2-9b": {
        "model_key": "gemma-2-9b-it",
        "model_name": "google/gemma-2-9b-it",
        "is_test": False,
    },

    # "gemma-2-27b": {
    #     "model_key": "gemma-2-27b-it",
    #     "model_name": "google/gemma-2-27b-it",
    #     "is_test": False,
    # },

    # --- Mistral ----
    "Mistral-7b-v0.3": {
        "model_key": "Mistral-7B-Instruct-v0.3",
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "is_test": False,
    },

    "Ministral-8b-2410": {
        "model_key": "Ministral-8B-Instruct-2410",
        "model_name": "mistralai/Ministral-8B-Instruct-2410",
        "is_test": False,
    },

    # "mistral-24b-2501": {
    #     "model_key": "Mistral-Small-24B-Instruct-2501",
    #     "model_name": "mistralai/Mistral-Small-24B-Instruct-2501",
    #     "is_test": False,
    # },
    # "Mistral32-24b-2506": {
    #     "model_key": "Mistral-Small-3.2-24B-Instruct-2506",
    #     "model_name": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    #     "is_test": False,
    # },
    
    # --- Qwen ---
    "Unsloth-Qwen3-1.7B": {
        "model_key": "Unsloth-Qwen3-1.7B",
        "model_name": "unsloth/Qwen3-1.7B",
        "is_test": False,
    },

    "Qwen3-1.7B": {
        "model_key": "Qwen3-1.7B",
        "model_name": "Qwen/Qwen3-1.7B",
        "is_test": False,
    },

    "Qwen3-4B": {
        "model_key": "Qwen3-4B",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "is_test": False,
    },

    "Qwen3-8B": {
        "model_key": "Qwen3-8B",
        "model_name": "Qwen/Qwen3-8B",
        "is_test": False,
    },
    
    
}

class ModelManager:
    """Manages model loading and task execution across multiple models."""
    
    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = outputs_dir
        self.model = None
        self.tokenizer = None
        self.current_model_key = None
        
        # Ensure outputs directory exists
        os.makedirs(self.outputs_dir, exist_ok=True)
    
    def load_model(self, model_name: str, model_key: str) -> bool:
        """Load model and tokenizer."""
        try:
            logging.info(f"Loading model '{model_name}'...")

            # Standard loading for all instruction models
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16
            )
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.padding_side = 'right'
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.pad_token is None and self.tokenizer.eos_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                self.model.resize_token_embeddings(len(self.tokenizer))

            self.current_model_key = model_key
            logging.info(f"Model and tokenizer loaded successfully.")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model '{model_name}'. Error: {e}")
            return False
    
    def unload_model(self):
        """Clean up model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer  
            self.tokenizer = None
        self.current_model_key = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Model unloaded and memory cleared.")
    
    def load_task_module(self, task_path: str):
        """Dynamically load a task module."""
        spec = importlib.util.spec_from_file_location("task_module", task_path)
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        return task_module
    
    def run_task(self, task_module, task_name: str, test_mode: bool = False) -> Optional[pd.DataFrame]:
        """Run a single task on the current model."""
        if self.model is None or self.tokenizer is None:
            logging.error("No model loaded. Cannot run task.")
            return None
        
        try:
            logging.info(f"Running task '{task_name}' on model '{self.current_model_key}'")
            
            # Run the task - pass model, tokenizer, and model_key to the task
            result_df = task_module.run_task(
                model=self.model,
                tokenizer=self.tokenizer,
                model_key=self.current_model_key,
                test_mode=test_mode
            )
            
            if result_df is not None and not result_df.empty:
                # Save results
                output_filename = f"{self.current_model_key}_{task_name}_results.csv"
                output_path = os.path.join(self.outputs_dir, output_filename)
                result_df.to_csv(output_path, index=False)
                logging.info(f"Task '{task_name}' completed. Results saved to {output_path}")
                return result_df
            else:
                logging.warning(f"Task '{task_name}' returned empty results.")
                return None
                
        except Exception as e:
            logging.error(f"Error running task '{task_name}' on model '{self.current_model_key}': {e}")
            return None
    
    def run_all_tasks_on_all_models(self, models_to_run: List[str], task_paths: List[str], test_mode: bool = False):
        """Main orchestrator: run all tasks on all models."""
        
        # Load all task modules
        task_modules = {}
        for task_path in task_paths:
            task_name = os.path.splitext(os.path.basename(task_path))[0]
            try:
                task_modules[task_name] = self.load_task_module(task_path)
                logging.info(f"Loaded task module: {task_name}")
            except Exception as e:
                logging.error(f"Failed to load task module '{task_path}': {e}")
                continue
        
        if not task_modules:
            logging.error("No task modules loaded. Exiting.")
            return
        
        # Process each model
        for model_key in models_to_run:
            if model_key not in MODEL_CONFIGS:
                logging.error(f"Invalid model key: {model_key}")
                continue
                
            model_config = MODEL_CONFIGS[model_key]
            logging.info(f"\n=== Processing model: {model_config['model_key']} ===")
            
            # Load the model
            if not self.load_model(model_config["model_name"], model_config["model_key"]):
                logging.error(f"Failed to load model {model_key}, skipping...")
                continue
            
            # Run all tasks on this model
            for task_name, task_module in task_modules.items():
                try:
                    self.run_task(task_module, task_name, test_mode)
                except Exception as e:
                    logging.error(f"Error running task {task_name} on model {model_key}: {e}")
                    continue
            
            # Unload model before moving to next one
            self.unload_model()
        
        logging.info("\n=== All models and tasks processed ===")

def main():
    parser = argparse.ArgumentParser(description="Run multiple tasks across multiple models.")
    parser.add_argument("--models", type=str, nargs="+", 
                        help="Space-separated list of specific models to run")
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()) + ["all"],
                        default="all", help="Single model to run (or 'all' for all models)")
    parser.add_argument("--tasks", type=str, nargs="+", required=True,
                        help="Space-separated list of task script paths")
    parser.add_argument("--outputs-dir", type=str, default="outputs",
                        help="Directory to save output files")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode (process fewer entries)")
    
    args = parser.parse_args()
    
    # Determine which models to run
    if args.models:
        models_to_run = args.models
    elif args.model == "all":
        models_to_run = list(MODEL_CONFIGS.keys())
    else:
        models_to_run = [args.model]
    
    # Validate model choices
    invalid_models = [m for m in models_to_run if m not in MODEL_CONFIGS]
    if invalid_models:
        logging.error(f"Invalid model(s): {invalid_models}")
        logging.error(f"Available models: {list(MODEL_CONFIGS.keys())}")
        return
    
    # Validate task files exist
    for task_path in args.tasks:
        if not os.path.exists(task_path):
            logging.error(f"Task file does not exist: {task_path}")
            return
    
    logging.info(f"Running evaluation on models: {models_to_run}")
    logging.info(f"Tasks to run: {[os.path.basename(t) for t in args.tasks]}")
    logging.info(f"Output directory: {args.outputs_dir}")
    logging.info(f"Test mode: {args.test}")
    
    # Initialize model manager and run everything
    manager = ModelManager(outputs_dir=args.outputs_dir)
    manager.run_all_tasks_on_all_models(models_to_run, args.tasks, test_mode=args.test)

if __name__ == "__main__":
    main()