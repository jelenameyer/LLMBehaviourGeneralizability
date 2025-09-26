#!/usr/bin/env python3
"""
Multi-model DOSPERT evaluation script.
Combines the model configurations from wa_scoring_batched.py with 
the DOSPERT candidate logprobs functionality.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import re
import argparse
import logging
import os
from typing import Tuple, Dict, List, Any

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

# --- File Paths & Settings ---
DATA_FILE = "survey_data/promptsDospertVaried.jsonl"
OUTPUTS_DIR = "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Globals ---
MODEL = None
TOKENIZER = None

def initialize_model_and_tokenizer(model_name: str) -> bool:
    """Load model+tokenizer with the same logic as the word association script."""
    global MODEL, TOKENIZER
    try:
        logging.info(f"Loading model '{model_name}'...")

        # Standard loading for all instruction models
        MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
        MODEL.eval()

        TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        TOKENIZER.padding_side = 'right'
        if TOKENIZER.pad_token is None and TOKENIZER.eos_token is not None:
            TOKENIZER.pad_token = TOKENIZER.eos_token
        elif TOKENIZER.pad_token is None and TOKENIZER.eos_token is None:
            TOKENIZER.add_special_tokens({"pad_token": "<|pad|>"})
            MODEL.resize_token_embeddings(len(TOKENIZER))

        # Note: All models are instruction-tuned and have chat templates

        logging.info(f"Model and tokenizer loaded successfully.")
        #logging.info(f"Chat template: {getattr(TOKENIZER, 'chat_template', 'None')}")
        return True
    except Exception as e:
        logging.error(f"Failed to load model '{model_name}'. Error: {e}")
        return False

def detect_chat_tokens(tokenizer: AutoTokenizer) -> Tuple[str, str]:
    """
    Returns (USER_TOK, ASSIST_TOK) automatically.
    Works with HF chat models using chat_template as a string.
    Falls back to <|user|> / <|assistant|>.
    """
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl and isinstance(tpl, str):
        # Look for first <|im_start|>role pattern
        user_match = re.search(r"<\|im_start\|>user", tpl)
        assist_match = re.search(r"<\|im_start\|>assistant", tpl)
        user_tok = user_match.group(0) if user_match else "<|user|>"
        assist_tok = assist_match.group(0) if assist_match else "<|assistant|>"
        return user_tok, assist_tok
    # Fallback defaults
    return "<|user|>", "<|assistant|>"

def has_chat_template(tokenizer: AutoTokenizer) -> bool:
    """Check if tokenizer has a chat template."""
    try:
        tmpl = getattr(tokenizer, "chat_template", None)
        return bool(tmpl)
    except Exception:
        return False

def candidate_logprobs_chatstyle(text: str, model_key: str) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts with logprobs for candidates 1–5
    at each << >> position, using chat template formatting.
    All models are instruction-tuned so they all use chat templates.
    """
    # Use chat template approach (all models have them)
    USER_TOK, ASSIST_TOK = detect_chat_tokens(TOKENIZER)
    #logging.info(f"Using chat template with USER_TOK={USER_TOK} ASSIST_TOK={ASSIST_TOK}")
    
    # Rebuild text into chat form
    lines = text.splitlines()
    rebuilt = []
    for ln in lines:
        m = re.match(r"(\d+)\.\s*(.*)<<(\d+)>>", ln)
        if m:
            qnum, qtext, ans = m.groups()
            rebuilt.append(f"{USER_TOK} {qnum}. {qtext} <<")
            rebuilt.append(f"{ASSIST_TOK} {ans.strip()}")
        else:
            rebuilt.append(ln)
    chat_text = "\n".join(rebuilt)
    
    # Encode and get logprobs
    enc = TOKENIZER(chat_text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = enc.input_ids.to(MODEL.device)
    offsets = enc.offset_mapping[0].tolist()
    
    # Find assistant responses
    pattern = re.compile(rf"{re.escape(ASSIST_TOK)}\s*(\d)")
    search_text = chat_text

    # Compute logprobs
    with torch.no_grad():
        out = MODEL(input_ids)
        logprobs = torch.nn.functional.log_softmax(out.logits, dim=-1)[0]

    # Candidate IDs 1–5
    try:
        cand_ids = [TOKENIZER.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]
    except IndexError:
        # Fallback if tokenizer doesn't have single-token numbers
        logging.warning("Could not get single-token IDs for 1-5, using multi-token encoding")
        cand_ids = [TOKENIZER.encode(str(i), add_special_tokens=False) for i in range(1, 6)]
        cand_ids = [ids[0] if ids else 0 for ids in cand_ids]  # Take first token of each

    results = []
    
    # Find all matches
    for m in pattern.finditer(search_text):
        human = m.group(1)
        span_lo, span_hi = m.span(1)
        
        # Find token index overlapping with the number
        try:
            tok_idx = next(
                i for i, (lo, hi) in enumerate(offsets)
                if not (hi <= span_lo or lo >= span_hi)
            )
            
            # Extract logprobs for candidates 1–5 at this position
            lp_candidates = {str(k): logprobs[tok_idx][cid].item()
                           for k, cid in zip(range(1, 6), cand_ids)}
            results.append(dict(human_number=human, **lp_candidates))
            
        except (StopIteration, IndexError) as e:
            logging.warning(f"Could not find token for span {span_lo}-{span_hi}: {e}")
            continue

    return results

def process_model(model_config: Dict[str, Any], data_file: str, test_mode: bool = False) -> None:
    """Process a single model configuration."""
    model_key = model_config["model_key"]
    model_name = model_config["model_name"]
    
    logging.info(f"=== Processing model: {model_key} ===")
    
    # Initialize model
    if not initialize_model_and_tokenizer(model_name):
        logging.error(f"Failed to initialize model {model_key}, skipping...")
        return
    
    # Setup output file
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUTS_DIR, f"{model_key}_dospert_scores.csv")
    
    # Process data
    all_rows = []
    
    try:
        with open(data_file) as f:
            entries = [json.loads(line) for line in f]
            
        # Limit entries in test mode
        if test_mode:
            entries = entries[:2]
            logging.info(f"Test mode: processing only first 2 entries")
            
        for entry_idx, entry in enumerate(entries):
            logging.info(f"Processing entry {entry_idx + 1}/{len(entries)}")
            
            try:
                spans = candidate_logprobs_chatstyle(entry["text"], model_key)
                
                for i, s in enumerate(spans, 1):
                    s["model"] = model_key
                    s["item"] = i
                    s["participant"] = entry["participant"]
                    s["flipped"] = entry.get("flipped", "")
                    s["experiment"] = entry.get("experiment", "")
                    all_rows.append(s)
                    
            except Exception as e:
                logging.error(f"Error processing entry {entry_idx}: {e}")
                continue
        
        # Save results
        if all_rows:
            df = pd.DataFrame(all_rows)
            df.to_csv(output_path, index=False)
            logging.info(f"Saved {len(all_rows)} results to {output_path}")
        else:
            logging.warning(f"No results to save for {model_key}")
            
    except Exception as e:
        logging.error(f"Error processing model {model_key}: {e}")
    
    finally:
        # Clean up model from memory
        global MODEL, TOKENIZER
        del MODEL, TOKENIZER
        MODEL, TOKENIZER = None, None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Run DOSPERT evaluation across multiple models.")
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()) + ["all"],
                        default="all", help="Model to run (or 'all' for all models)")
    parser.add_argument("--data-file", type=str, default=DATA_FILE,
                        help="Path to JSONL data file")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode (process fewer entries)")
    parser.add_argument("--models", type=str, nargs="+", 
                        help="Space-separated list of specific models to run")
    
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
    
    logging.info(f"Running evaluation on models: {models_to_run}")
    logging.info(f"Data file: {args.data_file}")
    logging.info(f"Test mode: {args.test}")
    
    # Process each model
    for model_key in models_to_run:
        try:
            model_config = MODEL_CONFIGS[model_key]
            process_model(model_config, args.data_file, test_mode=args.test)
        except Exception as e:
            logging.error(f"Failed to process model {model_key}: {e}")
            continue
    
    logging.info("=== All models processed ===")

if __name__ == "__main__":
    main()