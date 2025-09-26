
"""
BART (Baloon Analogue Risk Task) choice prediction scoring script for HuggingFaceTB/SmolLM2-1.7B-Instruct.
This script takes human decision-making data from experience-based choice tasks and calculates 
how likely the model finds the final choice decisions (x time to pump baloon or to stop) given the pumping and exploding history.
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
BATCH_SIZE = 32
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
        MODEL.eval()

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


def parse_bart_data(text: str) -> Dict[str, Any]:
    """
    Parse BART text and extract all information including balloon sequences and decision points.
    """
    # Extract pump and stop keys from instructions
    pump_match = re.search(r'pressing ([A-Z]) and you will accumulate', text)
    stop_match = re.search(r'pressing ([A-Z]) and you will collect', text)
    
    if not pump_match or not stop_match:
        raise ValueError("Could not identify pump and stop keys from instructions")
    
    pump_key = pump_match.group(1)
    stop_key = stop_match.group(1)
    #print(pump_key, stop_key)
    
    # Extract instructions
    instructions_end = text.find('\n\nBalloon 1:')
    instructions = text[:instructions_end].strip() if instructions_end != -1 else ""
    
    # Split into balloon sections
    balloon_pattern = r'\n\nBalloon (\d+):\n(.*?)(?=\n\nBalloon \d+:|\Z)'
    balloon_matches = re.findall(balloon_pattern, text, re.DOTALL)
    
    balloons = []
    for balloon_num, balloon_text in balloon_matches:
        # Parse key presses
        key_pattern = r'\{([' + re.escape(pump_key) + re.escape(stop_key) + r'])\}'
        key_presses = re.findall(key_pattern, balloon_text)
        
        # Determine outcome
        if "explodes" in balloon_text:
            outcome = "explode"
            final_score = 0
        else:
            score_match = re.search(r'get (\d+) points', balloon_text)
            outcome = "stop"
            final_score = int(score_match.group(1)) if score_match else 0
        
        balloons.append({
            "balloon_num": int(balloon_num),
            "key_presses": key_presses,
            "outcome": outcome,
            "final_score": final_score,
            "pump_count": len([k for k in key_presses if k == pump_key])
        })
    
    return {
        "instructions": instructions,
        "pump_key": pump_key,
        "stop_key": stop_key,
        "balloons": balloons
    }


def build_full_sequence_with_decisions(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a single sequence containing all balloons with decision points marked.
    This allows the model to learn from previous balloons.
    """
    instructions = parsed_data["instructions"]
    pump_key = parsed_data["pump_key"]
    stop_key = parsed_data["stop_key"]
    balloons = parsed_data["balloons"]
    
    # Build full text with all balloons
    full_text = instructions
    decision_points = []
    
    for balloon in balloons:
        balloon_num = balloon["balloon_num"]
        key_presses = balloon["key_presses"]
        outcome = balloon["outcome"]
        final_score = balloon["final_score"]
        
        # Add balloon header
        full_text += f"\n\nBalloon {balloon_num}:\nYou press "
        
        for i, key_press in enumerate(key_presses):
            # Rebuild prefix up to this decision
            prefix = full_text + " ".join([f"{{{k}}}" for k in key_presses[:i]]) 
            if i > 0:
                prefix += " "
            prefix += "{"
            
            prefix_tokens = TOKENIZER(prefix, add_special_tokens=False).input_ids
            
            decision_points.append({
                "balloon_num": balloon_num,
                "decision_num": i + 1,
                "pumps_so_far": key_presses[:i].count(pump_key),
                "choice_made": "pump" if key_press == pump_key else "stop",
                "balloon_outcome": outcome,
                "final_score": final_score,
                "prefix_length": len(prefix_tokens),   # NEW: store prefix length
                "prefix": prefix                      # optional: for debugging
            })
        
        # Complete the balloon text
        key_text = " ".join([f"{{{k}}}" for k in key_presses])
        full_text += key_text
        
        if outcome == "explode":
            full_text += ". The balloon was inflated too much and explodes."
        else:
            full_text += f". You stop inflating the balloon and get {final_score} points."
    
    # Tokenize the full sequence once
    device = MODEL.device
    input_ids = TOKENIZER(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    
    return {
        "input_ids": input_ids,
        "decision_points": decision_points,
        "pump_key": pump_key,
        "stop_key": stop_key
    }


def get_all_decision_logprobs(sequence_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get log probabilities for all decisions in one forward pass using prefix alignment (DFE-style).
    """
    input_ids = sequence_data["input_ids"]
    decision_points = sequence_data["decision_points"]
    pump_key = sequence_data["pump_key"]
    stop_key = sequence_data["stop_key"]
    
    # Single forward pass for the entire sequence
    with torch.inference_mode():
        outputs = MODEL(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)
    
    pump_token = TOKENIZER(pump_key, add_special_tokens=False).input_ids[0]
    stop_token = TOKENIZER(stop_key, add_special_tokens=False).input_ids[0]
    
    results = []
    for decision in decision_points:
        pos = decision["prefix_length"]
        pred_logits = logits[0, pos - 1, :]  # predict the token after prefix
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        
        results.append({
            "balloon_num": decision["balloon_num"],
            "decision_num": decision["decision_num"],
            "pumps_so_far": decision["pumps_so_far"],
            "human_decision": decision["choice_made"],
            "balloon_outcome": decision["balloon_outcome"],
            "final_score": decision["final_score"],
            "log_prob_pump": log_probs[pump_token].item(),
            "log_prob_stop": log_probs[stop_token].item(),
            "pump_key": pump_key,
            "stop_key": stop_key,
        })
    
    return results



def process_bart_participant(participant_data: dict, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Process a single BART participant using efficient sequence processing.
    """
    text = participant_data["text"]
    participant_id = participant_data["participant"]
    
    try:
        # Parse the data
        parsed_data = parse_bart_data(text)
        
        # Build sequence with decision points
        sequence_data = build_full_sequence_with_decisions(parsed_data)
        
        # Get predictions for all decisions in one pass
        decision_results = get_all_decision_logprobs(sequence_data)
        
        # Convert to standardized format
        standardized_results = []
        for result in decision_results:
            standardized_results.append({
                "model": MODEL_KEY,
                "task": "BART",
                "part_id": participant_id,
                "round": result["balloon_num"],  # balloon_num becomes round
                "decision_num": result["decision_num"],
                "human_decision": result["human_decision"],
                "log_prob_pump": result["log_prob_pump"],
                "log_prob_stop": result["log_prob_stop"],
                # Additional BART-specific info
                "pumps_so_far": result["pumps_so_far"],
                "balloon_outcome": result["balloon_outcome"],
                "final_score": result["final_score"],
                "pump_key": result["pump_key"],
                "stop_key": result["stop_key"],
                # Participant metadata
                "age": participant_data.get("age"),
                "sex": participant_data.get("sex"),
                "location": participant_data.get("location"),
            })
        
        if verbose:
            print(f"Participant {participant_id}: {len(decision_results)} decisions across {len(parsed_data['balloons'])} balloons")
        
        return standardized_results
        
    except Exception as e:
        logging.error(f"Error processing participant {participant_id}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Score BART decisions with SmolLM2-1.7B-Instruct using efficient masking.")
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
    output_path = os.path.join(OUTPUTS_DIR, f"{MODEL_KEY}_bart_results.csv")

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
    start_time = time.time()
    
    for i, participant in enumerate(participant_data):
        logging.info(f"Processing participant {i+1}/{len(participant_data)}: {participant.get('participant', 'unknown')}")
        
        try:
            results = process_bart_participant(participant, verbose=args.verbose)
            all_results.extend(results)
        except Exception as e:
            logging.error(f"Error processing participant {participant.get('participant', 'unknown')}: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path, index=False)
        logging.info(f"Saved {len(all_results)} decision predictions to {output_path}")
        
        # Summary statistics
        print(f"\n--- SUMMARY ---")
        print(f"Processed {len(participant_data)} participants in {elapsed_time:.1f} seconds")
        print(f"Average {elapsed_time/len(participant_data):.2f} seconds per participant")
        print(f"Total decisions: {len(all_results)}")
        print(f"Average decisions per participant: {len(all_results) / len(participant_data):.1f}")
        
        if len(all_results) > 0:
            print(f"\nChoice distribution:")
            print(results_df['human_decision'].value_counts())
            
            print(f"\nBalloon outcome distribution:")
            print(results_df.groupby(['part_id', 'round'])['balloon_outcome'].first().value_counts())
            
            avg_pumps = results_df.groupby(['part_id', 'round'])['pumps_so_far'].max().mean()
            print(f"\nAverage pumps per balloon: {avg_pumps:.1f}")
            
            # Show model efficiency
            total_forward_passes = len(participant_data)  # One per participant
            total_decisions = len(all_results)
            print(f"\nEfficiency: {total_forward_passes} forward passes for {total_decisions} decisions")
            print(f"({total_decisions/total_forward_passes:.0f} decisions per forward pass)")
    else:
        logging.warning("No results to save.")

if __name__ == "__main__":
    main()