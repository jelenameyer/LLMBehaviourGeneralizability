"""
DFE (Decisions From Experience) choice prediction scoring script for HuggingFaceTB/SmolLM2-1.7B-Instruct.
This script takes human decision-making data from experience-based choice tasks and calculates 
how likely the model finds the final choice decisions (Box A vs Box B) given the sampling history.
"""

import os
import time
from datetime import timedelta
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Dict, List, Any, Optional, Tuple
import argparse
import json
import re

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Settings ---
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MODEL_KEY = "SmolLM2-1.7B-Instruct"
OUTPUTS_DIR = "outputs"
BATCH_SIZE = 32  # Small batch size since DFE texts are long
TEST_MAX_ROWS = 3

# --- Globals ---
MODEL = None
TOKENIZER = None

def _has_chat_template(tok: AutoTokenizer) -> bool:
    """Check if tokenizer has a chat template."""
    try:
        tmpl = getattr(tok, "chat_template", None)
        return bool(tmpl)
    except Exception:
        return False

def _get_eot_id(tok: AutoTokenizer) -> Optional[int]:
    """Get end-of-turn token ID, fallback to EOS if available."""
    try:
        eid = tok.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eid, int) and eid >= 0:
            return eid
    except Exception:
        pass
    return tok.eos_token_id

def initialize_model_and_tokenizer(model_name: str = None) -> bool:
    """Load the model and tokenizer."""
    global MODEL, TOKENIZER
    model_name = model_name or MODEL_NAME
    
    try:
        logging.info(f"Loading model '{model_name}'...")
        
        MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None, #"auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
        MODEL.eval() # run model in evaluation (not training) mode

        TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        TOKENIZER.padding_side = 'right'
        
        # Set up padding token
        if TOKENIZER.pad_token is None and TOKENIZER.eos_token is not None:
            TOKENIZER.pad_token = TOKENIZER.eos_token
        elif TOKENIZER.pad_token is None and TOKENIZER.eos_token is None:
            TOKENIZER.add_special_tokens({"pad_token": "<|pad|>"})
            MODEL.resize_token_embeddings(len(TOKENIZER))

        logging.info("Model and tokenizer loaded successfully. Chat template detected: %s",
                     _has_chat_template(TOKENIZER))
        return True
    except Exception as e:
        logging.error(f"Failed to load model '{model_name}'. Error: {e}")
        return False

def extract_box_labels(instructions: str) -> Tuple[str, str]:
    """
    Extract the randomized box labels (e.g., 'K' and 'M') from the session instructions.
    """
    match = re.search(r'presented with two boxes: Box ([A-Z]) and Box ([A-Z])', instructions)
    if not match:
        raise ValueError("Could not extract box labels from instructions")
    print(match.group(1), match.group(2))
    return match.group(1), match.group(2)

def extract_human_decision(round_text: str) -> Optional[str]:
    """Extract the actual human decision (A or B) from a round."""
    # Look for pattern: "You decided to <<choose>> Box <<A>>" or similar
    match = re.search(r'You decided to <<choose>> Box <<([A-Z])>>', round_text)
    if match:
        return match.group(1)
    return None

def _build_dfe_full_sequence(rounds: List[str], box_labels: Tuple[str, str]) -> Dict[str, Any]:
    """
    Build a single tokenized sequence for all rounds of one participant.
    Handles randomized box labels (e.g., K vs. M).
    """
    device = MODEL.device
    pad_id = TOKENIZER.pad_token_id or TOKENIZER.eos_token_id

    full_text = ""
    choice_positions = []
    human_decisions = []

    for i, r in enumerate(rounds, 1):
        full_text += f"\n\nProblem {i}:\n{r}\n"

        # Extract human decision (any letter)
        human_decision = extract_human_decision(r)
        human_decisions.append(human_decision)

        # Where to stop before the actual choice
        prefix = full_text.rsplit("You decided to <<choose>> Box <<", 1)[0] + "You decided to <<choose>> Box <<"
        prefix_ids = TOKENIZER(prefix, add_special_tokens=False).input_ids

        choice_positions.append(len(prefix_ids))

    # tokenize full text
    full_ids = TOKENIZER(full_text, add_special_tokens=False).input_ids
    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)

    return {
        "input_ids": input_ids,
        "choice_positions": choice_positions,
        "human_decisions": human_decisions,
        "box_labels": box_labels,  # keep for later
    }


def get_choice_logprobs_full(rounds: List[str], instructions: str) -> List[Dict[str, Any]]:
    # detect this participant’s labels
    box_labels = extract_box_labels(instructions)

    pack = _build_dfe_full_sequence(rounds, box_labels)
    input_ids = pack["input_ids"]
    choice_positions = pack["choice_positions"]
    human_decisions = pack["human_decisions"]
    box_labels = pack["box_labels"]

    with torch.inference_mode():
        outputs = MODEL(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)

    # get token ids for the participant’s box letters
    token_first = TOKENIZER(box_labels[0], add_special_tokens=False).input_ids[0]
    token_second = TOKENIZER(box_labels[1], add_special_tokens=False).input_ids[0]

    results = []
    for round_idx, (pos, human_choice) in enumerate(zip(choice_positions, human_decisions)):
        pred_logits = logits[0, pos - 1, :]  # model predicts token at 'pos'
        log_probs = torch.log_softmax(pred_logits, dim=-1)

        results.append({
            "round": round_idx + 1,
            "human_decision": human_choice,
            "log_prob_A": log_probs[token_first].item(),
            "log_prob_B": log_probs[token_second].item(),
        })

    return results


def process_dfe_participant(participant_data: dict, model_key: str, verbose: bool = False) -> List[Dict[str, Any]]:
    text = participant_data["text"]
    participant_id = participant_data["participant"]

    # Split into 8 problem sections
    rounds = re.split(r'\n\nProblem \d+:\n', text)[1:]  # drop instructions

    if not rounds:
        logging.warning(f"No problems found for participant {participant_id}")
        return []

    # Get log probs for both A and B at each round
    choice_probs = get_choice_logprobs_full(rounds, text)

    # Add metadata in standardized format
    standardized_results = []
    for result in choice_probs:
        standardized_results.append({
            "model": model_key,
            "task": "DFE",
            "part_id": participant_id,
            "round": result["round"],
            "decision_num": None,  # DFE doesn't have multiple decisions per round
            "human_decision": result["human_decision"],
            "log_prob_A": result["log_prob_A"],
            "log_prob_B": result["log_prob_B"],
            # Additional metadata
            "age": participant_data.get("age"),
            "sex": participant_data.get("sex"),
            "location": participant_data.get("location"),
        })

    return standardized_results

def run_dfe_scoring(input_file: str, model_name: str = None, model_key: str = None, 
                   test_mode: bool = False, verbose: bool = False) -> pd.DataFrame:
    """Main function to run DFE scoring - can be called from external scripts."""
    
    # Use provided model or initialize new one
    if MODEL is None:
        if not initialize_model_and_tokenizer(model_name):
            raise RuntimeError("Failed to initialize model")
    
    # Determine model key
    if model_key is None:
        model_key = model_name.split('/')[-1] if model_name else MODEL_KEY

    # Load participant data
    logging.info(f"Loading DFE data from {input_file}...")
    
    if input_file.endswith('.json'):
        with open(input_file, 'r') as f:
            participant_data = json.load(f)
    elif input_file.endswith('.jsonl'):
        participant_data = []
        with open(input_file, 'r') as f:
            for line in f:
                participant_data.append(json.loads(line.strip()))
    else:
        raise ValueError("Input file must be .json or .jsonl format")

    # Handle single participant vs list of participants
    if isinstance(participant_data, dict):
        participant_data = [participant_data]
    
    if test_mode:
        participant_data = participant_data[:TEST_MAX_ROWS]
        logging.info(f"Test mode: Processing only {len(participant_data)} DFE participants.")

    # Process all participants
    all_results = []
    
    for i, participant in enumerate(participant_data):
        logging.info(f"Processing DFE participant {i+1}/{len(participant_data)}: {participant.get('participant', 'unknown')}")
        
        try:
            results = process_dfe_participant(participant, model_key, verbose=verbose)
            all_results.extend(results)
        except Exception as e:
            logging.error(f"Error processing DFE participant {participant.get('participant', 'unknown')}: {e}")
            continue
    
    return pd.DataFrame(all_results)

def main():
    parser = argparse.ArgumentParser(description="Score DFE choice decisions with SmolLM2-1.7B-Instruct.")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON file with participant data")
    parser.add_argument("--test", action="store_true",
                        help=f"Run in test mode (process only {TEST_MAX_ROWS} participants).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output.")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"Model name to use (default: {MODEL_NAME}).")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        return

    # Set up output
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    model_key = args.model.split('/')[-1]
    output_path = os.path.join(OUTPUTS_DIR, f"{model_key}_dfe_results.csv")

    # Run scoring
    try:
        results_df = run_dfe_scoring(args.input, args.model, model_key, args.test, args.verbose)
        
        if not results_df.empty:
            results_df.to_csv(output_path, index=False)
            logging.info(f"Saved {len(results_df)} DFE choice predictions to {output_path}")
            
            # Summary statistics
            print(f"\n--- DFE SUMMARY ---")
            print(f"Processed {results_df['part_id'].nunique()} participants")
            print(f"Total choice predictions: {len(results_df)}")
            print(f"Average problems per participant: {len(results_df) / results_df['part_id'].nunique():.1f}")
            
            if args.verbose and 'human_decision' in results_df.columns:
                print(f"\nHuman choice distribution:")
                print(results_df['human_decision'].value_counts())
        else:
            logging.warning("No DFE results to save.")
            
    except Exception as e:
        logging.error(f"DFE scoring failed: {e}")

if __name__ == "__main__":
    main()