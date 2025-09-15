
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
TEST_MAX_ROWS = 5

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

def initialize_model_and_tokenizer() -> bool:
    """Load the SmolLM model and tokenizer."""
    global MODEL, TOKENIZER
    try:
        logging.info(f"Loading model '{MODEL_NAME}'...")
        
        MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
        MODEL.eval() # run model in evaluation (not training) mode

        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
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
        logging.error(f"Failed to load model '{MODEL_NAME}'. Error: {e}")
        return False


def _build_dfe_full_sequence(rounds: List[str]) -> Dict[str, Any]:
    """
    Build a single tokenized sequence for all rounds of one participant.
    We will later extract logprobs for 'A' and 'B' at each choice point.
    """
    device = MODEL.device
    pad_id = TOKENIZER.pad_token_id or TOKENIZER.eos_token_id
    
    full_text = ""
    choice_positions = []
    
    for i, r in enumerate(rounds, 1):
        full_text += f"\n\nProblem {i}:\n{r}\n"
        
        # Where to stop before the actual choice
        prefix = full_text.rsplit("You decided to <<choose>> Box <<", 1)[0] + "You decided to <<choose>> Box <<"
        prefix_ids = TOKENIZER(prefix, add_special_tokens=False).input_ids
        
        # record position where model will predict A or B
        choice_positions.append(len(prefix_ids))
    
    # tokenize full text (with dummy endings, doesn’t matter)
    full_ids = TOKENIZER(full_text, add_special_tokens=False).input_ids
    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    return {
        "input_ids": input_ids,
        "choice_positions": choice_positions,
    }


def get_choice_logprobs_full(rounds: List[str]) -> List[Dict[str, Any]]:
    pack = _build_dfe_full_sequence(rounds)
    input_ids = pack["input_ids"]
    choice_positions = pack["choice_positions"]

    with torch.inference_mode():
        outputs = MODEL(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)

    results = []
    for round_idx, pos in enumerate(choice_positions):
        pred_logits = logits[0, pos - 1, :]  # model predicts token at 'pos'
        log_probs = torch.log_softmax(pred_logits, dim=-1)

        token_A = TOKENIZER("A", add_special_tokens=False).input_ids[0]
        token_B = TOKENIZER("B", add_special_tokens=False).input_ids[0]

        results.append({
            "round": round_idx + 1,
            "log_prob_A": log_probs[token_A].item(),
            "log_prob_B": log_probs[token_B].item(),
        })

    return results


def process_dfe_participant(participant_data: dict, verbose: bool = False) -> List[Dict[str, Any]]:
    text = participant_data["text"]
    participant_id = participant_data["participant"]

    # Split into 8 problem sections
    rounds = re.split(r'\n\nProblem \d+:\n', text)[1:]  # drop instructions

    if not rounds:
        logging.warning(f"No problems found for participant {participant_id}")
        return []

    # Get log probs for both A and B at each round
    choice_probs = get_choice_logprobs_full(rounds)

    # Add metadata
    for result in choice_probs:
        result.update({
            "participant_id": participant_id,
            "model": MODEL_KEY,
            "age": participant_data.get("age"),
            "sex": participant_data.get("sex"),
            "location": participant_data.get("location"),
        })

    return choice_probs


def main():
    parser = argparse.ArgumentParser(description="Score DFE choice decisions with SmolLM2-1.7B-Instruct.")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON file with participant data")
    parser.add_argument("--test", action="store_true",
                        help=f"Run in test mode (process only {TEST_MAX_ROWS} participants).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for processing (default: {BATCH_SIZE}).")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        return

    # Set up output
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUTS_DIR, f"{MODEL_KEY}_dfe_results.csv")

    # Initialize model
    if not initialize_model_and_tokenizer():
        return

    # Load participant data
    logging.info(f"Loading data from {args.input}...")
    
    if args.input.endswith('.json'):
        with open(args.input, 'r') as f:
            participant_data = json.load(f)
    elif args.input.endswith('.jsonl'):
        participant_data = []
        with open(args.input, 'r') as f:
            for line in f:
                participant_data.append(json.loads(line.strip()))
    else:
        logging.error("Input file must be .json or .jsonl format")
        return

    # Handle single participant vs list of participants
    if isinstance(participant_data, dict):
        participant_data = [participant_data]
    
    if args.test:
        participant_data = participant_data[:TEST_MAX_ROWS]
        logging.info(f"Test mode: Processing only {len(participant_data)} participants.")

    # Process all participants
    all_results = []
    
    for i, participant in enumerate(participant_data):
        logging.info(f"Processing participant {i+1}/{len(participant_data)}: {participant.get('participant', 'unknown')}")
        
        try:
            results = process_dfe_participant(participant, verbose=args.verbose)
            all_results.extend(results)
        except Exception as e:
            logging.error(f"Error processing participant {participant.get('participant', 'unknown')}: {e}")
            continue
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path, index=False)
        logging.info(f"Saved {len(all_results)} choice predictions to {output_path}")
        
        # Summary statistics
        print(f"\n--- SUMMARY ---")
        print(f"Processed {len(participant_data)} participants")
        print(f"Total choice predictions: {len(all_results)}")
        print(f"Average problems per participant: {len(all_results) / len(participant_data):.1f}")
        
        if args.verbose:
            print(f"\nChoice distribution:")
            print(results_df['choice'].value_counts())
    else:
        logging.warning("No results to save.")

if __name__ == "__main__":
    main()