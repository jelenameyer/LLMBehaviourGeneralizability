"""
Example script showing how to use the multi-task cognitive scoring system.
"""

from multi_task_scorer import MultiTaskScorer, batch_run_models, AVAILABLE_MODELS

def example_single_model():
    """Example: Run all tasks with a single model."""
    print("=== Single Model Example ===")
    
    # Initialize scorer with a specific model
    scorer = MultiTaskScorer("HuggingFaceTB/SmolLM2-1.7B-Instruct", output_dir="example_outputs")
    
    # Initialize the model once
    if not scorer.initialize_model():
        print("Failed to initialize model")
        return
    
    # Run individual tasks
    dfe_results = scorer.run_task("dfe", "data/dfe_data.json", test_mode=True, verbose=True)
    bart_results = scorer.run_task("bart", "data/bart_data.json", test_mode=True, verbose=True)
    
    # Save combined results
    combined_file = scorer.save_combined_results()
    print(f"Combined results saved to: {combined_file}")
    
    # Clean up
    scorer.cleanup()

def example_multiple_models():
    """Example: Run multiple models on all tasks."""
    print("=== Multiple Models Example ===")
    
    models_to_test = ["smollm2-1.7b", "mistral-7b"]  # Add more as needed
    
    results_files = batch_run_models(
        models=models_to_test,
        input_dir="data",
        output_dir="batch_outputs", 
        test_mode=True  # Use test mode for quick testing
    )
    
    print("\nResults summary:")
    for model, file_path in results_files.items():
        print(f"{model}: {file_path}")

def example_specific_tasks():
    """Example: Run only specific tasks."""
    print("=== Specific Tasks Example ===")
    
    scorer = MultiTaskScorer("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    
    # Run only DFE task
    scorer.run_all_tasks(
        input_dir="data",
        tasks=["dfe"],  # Only run DFE
        test_mode=True,
        verbose=True
    )
    
    scorer.save_combined_results()
    scorer.cleanup()

def show_available_options():
    """Show what models and tasks are available."""
    print("=== Available Options ===")
    
    print("Available models:")
    for key, full_name in AVAILABLE_MODELS.items():
        print(f"  {key}: {full_name}")
    
    print("\nAvailable tasks:")
    from multi_task_scorer import TASK_CONFIGS
    for task_key, config in TASK_CONFIGS.items():
        print(f"  {task_key}: {config['name']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_type = sys.argv[1]
        
        if example_type == "single":
            example_single_model()
        elif example_type == "multiple":
            example_multiple_models()
        elif example_type == "tasks":
            example_specific_tasks()
        elif example_type == "options":
            show_available_options()
        else:
            print("Unknown example type. Options: single, multiple, tasks, options")
    else:
        print("Usage examples:")
        print("python example_usage.py single    # Run single model example")
        print("python example_usage.py multiple  # Run multiple models example") 
        print("python example_usage.py tasks     # Run specific tasks example")
        print("python example_usage.py options   # Show available options")
        print("\nOr run the main multi-task scorer directly:")
        print("python multi_task_scorer.py --model smollm2-1.7b --input-dir data --test --verbose")