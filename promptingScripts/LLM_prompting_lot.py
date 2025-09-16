"""
LOT (Lottery Task) choice prediction scoring script for HuggingFaceTB/SmolLM2-1.7B-Instruct.
This script takes human decision-making data from lottery choice tasks and calculates 
how likely the model finds the final choice decisions (A or B) given the lottery options.
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


def parse_lot_data(text: str) -> Dict[str, Any]:
    """
    Parse LOT text and extract all lottery problems and choices.
    """
    # Extract instructions (everything before first "Problem")
    instructions_end = text.find('\nProblem 1:')
    if instructions_end == -1:
        instructions_end = text.find('\n\nProblem 1:')
    instructions = text[:instructions_end].strip() if instructions_end != -1 else ""
    
    # Split into problem sections
    # Handle both single and double newlines before "Problem"
    problem_pattern = r'\n\n?Problem (\d+):\n(.*?)(?=\n\n?Problem \d+:|\Z)'
    problem_matches = re.findall(problem_pattern, text, re.DOTALL)
    
    problems = []
    for problem_num, problem_text in problem_matches:
        # Parse lottery pairs within this problem
        lottery_pairs = []
        
        # Split by double newlines to get individual lottery comparisons
        comparisons = re.split(r'\n\n', problem_text.strip())
        
        for comparison in comparisons:
            if not comparison.strip():
                continue
                
            # Parse lottery A and B options
            lottery_a_match = re.search(r'Lottery A: (.+?)(?=\nLottery B:)', comparison, re.DOTALL)
            lottery_b_match = re.search(r'Lottery B: (.+?)(?=\nYou chose)', comparison, re.DOTALL)
            choice_match = re.search(r'You chose <<([AB])>>', comparison)
            
            if lottery_a_match and lottery_b_match and choice_match:
                lottery_pairs.append({
                    "lottery_a": lottery_a_match.group(1).strip(),
                    "lottery_b": lottery_b_match.group(1).strip(),
                    "choice_made": choice_match.group(1)
                })
        
        problems.append({
            "problem_num": int(problem_num),
            "lottery_pairs": lottery_pairs
        })
    
    return {
        "instructions": instructions,
        "problems": problems
    }


def build_full_sequence_with_decisions(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a single sequence containing all problems with decision points marked.
    This allows the model to learn from previous lottery choices.
    """
    instructions = parsed_data["instructions"]
    problems = parsed_data["problems"]
    
    # Build full text with all problems
    full_text = instructions
    decision_points = []
    
    for problem in problems:
        problem_num = problem["problem_num"]
        lottery_pairs = problem["lottery_pairs"]
        
        # Add problem header
        full_text += f"\n\nProblem {problem_num}:"
        
        for pair_idx, pair in enumerate(lottery_pairs):
            lottery_a = pair["lottery_a"]
            lottery_b = pair["lottery_b"]
            choice_made = pair["choice_made"]
            
            # Add lottery options
            full_text += f"\nLottery A: {lottery_a}"
            full_text += f"\nLottery B: {lottery_b}"
            full_text += "\nYou chose <<"
            
            # Mark decision point (where model will predict A or B)
            prefix_text = full_text
            
            # Store decision information
            decision_points.append({
                "problem_num": problem_num,
                "pair_num": pair_idx + 1,
                "lottery_a": lottery_a,
                "lottery_b": lottery_b,
                "choice_made": choice_made,
                "token_position": None  # Will be filled after tokenization
            })
            
            # Complete the choice
            full_text += f"{choice_made}>>."
            
            # Add spacing if not the last pair in problem
            if pair_idx < len(lottery_pairs) - 1:
                full_text += "\n"
    
    # Tokenize the full sequence once
    device = MODEL.device
    input_ids = TOKENIZER(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    
    # Find token positions for each decision point
    updated_decision_points = []
    
    for decision in decision_points:
        problem_num = decision["problem_num"]
        pair_num = decision["pair_num"]
        
        # Rebuild prefix up to this decision
        prefix = instructions
        
        # Add all previous problems
        for prev_problem in problems:
            if prev_problem["problem_num"] > problem_num:
                break
            elif prev_problem["problem_num"] < problem_num:
                # Add complete previous problem
                prefix += f"\n\nProblem {prev_problem['problem_num']}:"
                for pair in prev_problem["lottery_pairs"]:
                    prefix += f"\nLottery A: {pair['lottery_a']}"
                    prefix += f"\nLottery B: {pair['lottery_b']}"
                    prefix += f"\nYou chose <<{pair['choice_made']}>>."
                    if pair != prev_problem["lottery_pairs"][-1]:
                        prefix += "\n"
            else:
                # Current problem - add up to current pair
                prefix += f"\n\nProblem {problem_num}:"
                current_problem = prev_problem  # This is the current problem
                
                for i, pair in enumerate(current_problem["lottery_pairs"]):
                    if i + 1 > pair_num:
                        break
                    elif i + 1 < pair_num:
                        # Complete previous pairs
                        prefix += f"\nLottery A: {pair['lottery_a']}"
                        prefix += f"\nLottery B: {pair['lottery_b']}"
                        prefix += f"\nYou chose <<{pair['choice_made']}>>.\n"
                    else:
                        # Current pair - up to decision point
                        prefix += f"\nLottery A: {pair['lottery_a']}"
                        prefix += f"\nLottery B: {pair['lottery_b']}"
                        prefix += "\nYou chose <<"
        
        # Tokenize prefix to find position
        prefix_tokens = TOKENIZER(prefix, add_special_tokens=False).input_ids
        decision["token_position"] = len(prefix_tokens)
        
        updated_decision_points.append(decision)
    
    return {
        "input_ids": input_ids,
        "decision_points": updated_decision_points
    }


def get_all_decision_logprobs(sequence_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get log probabilities for all decisions in one forward pass using masking.
    """
    input_ids = sequence_data["input_ids"]
    decision_points = sequence_data["decision_points"]
    
    # Single forward pass for the entire sequence
    with torch.inference_mode():
        outputs = MODEL(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)
    
    # Get token IDs for A and B
    a_token = TOKENIZER("A", add_special_tokens=False).input_ids[0]
    b_token = TOKENIZER("B", add_special_tokens=False).input_ids[0]
    
    results = []
    
    for decision in decision_points:
        pos = decision["token_position"]
        
        # Get logits at the decision position (predict next token after "<<")
        pred_logits = logits[0, pos - 1, :]  # -1 because we predict the next token
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        
        result = {
            "problem_num": decision["problem_num"],
            "pair_num": decision["pair_num"],
            "lottery_a": decision["lottery_a"],
            "lottery_b": decision["lottery_b"],
            "choice_made": decision["choice_made"],
            "log_prob_a": log_probs[a_token].item(),
            "log_prob_b": log_probs[b_token].item(),
        }
        
        results.append(result)
    
    return results


def process_lot_participant(participant_data: dict, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Process a single LOT participant using efficient sequence processing.
    """
    text = participant_data["text"]
    participant_id = participant_data["participant"]
    
    try:
        # Parse the data
        parsed_data = parse_lot_data(text)
        
        # Build sequence with decision points
        sequence_data = build_full_sequence_with_decisions(parsed_data)
        
        # Get predictions for all decisions in one pass
        decision_results = get_all_decision_logprobs(sequence_data)
        
        # Add participant metadata
        for result in decision_results:
            result.update({
                "participant_id": participant_id,
                "model": MODEL_KEY,
                "age": participant_data.get("age"),
                "sex": participant_data.get("sex"),
                "location": participant_data.get("location"),
                "experiment": participant_data.get("experiment"),
            })
        
        if verbose:
            total_problems = len(parsed_data['problems'])
            print(f"Participant {participant_id}: {len(decision_results)} decisions across {total_problems} problems")
        
        return decision_results
        
    except Exception as e:
        logging.error(f"Error processing participant {participant_id}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Score LOT decisions with SmolLM2-1.7B-Instruct using efficient masking.")
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
    output_path = os.path.join(OUTPUTS_DIR, f"{MODEL_KEY}_lot_results.csv")

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
            results = process_lot_participant(participant, verbose=args.verbose)
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
            print(results_df['choice_made'].value_counts())
            
            print(f"\nProblems processed:")
            print(f"Min problem: {results_df['problem_num'].min()}")
            print(f"Max problem: {results_df['problem_num'].max()}")
            
            # Show model efficiency
            total_forward_passes = len(participant_data)  # One per participant
            total_decisions = len(all_results)
            print(f"\nEfficiency: {total_forward_passes} forward passes for {total_decisions} decisions")
            print(f"({total_decisions/total_forward_passes:.0f} decisions per forward pass)")
            
            # Show sample results
            print(f"\nSample results:")
            sample_df = results_df.head(3)[['participant_id', 'problem_num', 'pair_num', 'choice_made', 'log_prob_a', 'log_prob_b']]
            for _, row in sample_df.iterrows():
                prob_a = torch.exp(torch.tensor(row['log_prob_a'])).item()
                prob_b = torch.exp(torch.tensor(row['log_prob_b'])).item()
                print(f"  P{row['participant_id']} Prob{row['problem_num']}.{row['pair_num']}: chose {row['choice_made']}, "
                      f"P(A)={prob_a:.3f}, P(B)={prob_b:.3f}")
    else:
        logging.warning("No results to save.")

if __name__ == "__main__":
    main()