#!/usr/bin/env python3
"""
BART Task Module - Balloon Analogue Risk Task evaluation WITHOUT chat template.
This module can be imported and run by the model manager.
Uses the original prefix-based approach for comparison with chat template results.
"""

import json
import torch
import pandas as pd
import re
import logging
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# Task-specific configuration
DATA_FILE = "bart_data/prompts_bart.jsonl"  # Update with your BART data file

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

def build_full_sequence_with_decisions(parsed_data: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Build a single sequence containing all balloons with decision points marked.
    This allows the model to learn from previous balloons.
    Uses the original prefix-based approach (no chat template).
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
            
            prefix_tokens = tokenizer(prefix, add_special_tokens=False).input_ids
            
            decision_points.append({
                "balloon_num": balloon_num,
                "decision_num": i + 1,
                "pumps_so_far": key_presses[:i].count(pump_key),
                "choice_made": "pump" if key_press == pump_key else "stop",
                "balloon_outcome": outcome,
                "final_score": final_score,
                "prefix_length": len(prefix_tokens),   # Store prefix length
                "prefix": prefix                      # Optional: for debugging
            })
        
        # Complete the balloon text
        key_text = " ".join([f"{{{k}}}" for k in key_presses])
        full_text += key_text
        
        if outcome == "explode":
            full_text += ". The balloon was inflated too much and explodes."
        else:
            full_text += f". You stop inflating the balloon and get {final_score} points."
    
    # Tokenize the full sequence once
    input_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids
    
    return {
        "input_ids": input_ids,
        "decision_points": decision_points,
        "pump_key": pump_key,
        "stop_key": stop_key,
        "full_text": full_text,  # For debugging
        "chat_format": False  # Indicate this is NOT chat format
    }

def get_all_decision_logprobs(sequence_data: Dict[str, Any], model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> List[Dict[str, Any]]:
    """
    Get log probabilities for all decisions in one forward pass using prefix alignment (DFE-style).
    Original approach without chat template.
    """
    input_ids = sequence_data["input_ids"]
    decision_points = sequence_data["decision_points"]
    pump_key = sequence_data["pump_key"]
    stop_key = sequence_data["stop_key"]
    
    # Single forward pass for the entire sequence
    with torch.inference_mode():
        outputs = model(input_ids=input_ids.to(model.device))
        logits = outputs.logits  # (1, seq_len, vocab)
    
    pump_token = tokenizer(pump_key, add_special_tokens=False).input_ids[0]
    stop_token = tokenizer(stop_key, add_special_tokens=False).input_ids[0]
    
    results = []
    for decision in decision_points:
        try:
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
        except Exception as e:
            logging.error(f"Error processing decision for balloon {decision['balloon_num']}, decision {decision['decision_num']}: {e}")
            continue
    
    return results

def process_bart_participant(participant_data: dict, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_key: str, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Process a single BART participant using efficient sequence processing.
    Uses original prefix-based approach (no chat template).
    """
    text = participant_data["text"]
    participant_id = participant_data["participant"]
    
    try:
        # Parse the data
        parsed_data = parse_bart_data(text)
        
        if verbose:
            logging.info(f"Participant {participant_id}: Found {len(parsed_data['balloons'])} balloons")
            total_decisions = sum(len(b['key_presses']) for b in parsed_data['balloons'])
            logging.info(f"Participant {participant_id}: Total decisions to process: {total_decisions}")
        
        # Build sequence with decision points
        sequence_data = build_full_sequence_with_decisions(parsed_data, tokenizer)
        
        if verbose:
            logging.info(f"Participant {participant_id}: Built {len(sequence_data['decision_points'])} decision points")
        
        # Get predictions for all decisions in one pass
        decision_results = get_all_decision_logprobs(sequence_data, model, tokenizer)
        
        if verbose:
            logging.info(f"Participant {participant_id}: Successfully processed {len(decision_results)} decisions")
        
        # Convert to standardized format
        standardized_results = []
        for result in decision_results:
            standardized_results.append({
                "model": model_key,
                "task": "BART_NoChat",  # Distinguish from chat version
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
        
        return standardized_results
        
    except Exception as e:
        logging.error(f"Error processing participant {participant_id}: {e}")
        return []

def run_task(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_key: str, 
             test_mode: bool = False, data_file: str = DATA_FILE) -> pd.DataFrame:
    """
    Main task runner function for BART evaluation without chat template.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        model_key: String identifier for the model
        test_mode: Whether to run in test mode (fewer entries)
        data_file: Path to the data file
    
    Returns:
        pandas.DataFrame: Results dataframe with BART decision logprobs
    """
    logging.info(f"Starting BART NoChat task for model: {model_key}")
    
    all_results = []
    
    try:
        # Load participant data
        if data_file.endswith('.json'):
            with open(data_file, 'r') as f:
                participant_data = json.load(f)
        elif data_file.endswith('.jsonl'):
            participant_data = []
            with open(data_file, 'r') as f:
                for line in f:
                    participant_data.append(json.loads(line.strip()))
        else:
            logging.error("Data file must be .json or .jsonl format")
            return pd.DataFrame()
        
        # Handle single participant vs list of participants
        if isinstance(participant_data, dict):
            participant_data = [participant_data]
            
        # Limit entries in test mode
        if test_mode:
            participant_data = participant_data[:10]
            logging.info(f"Test mode: processing only first 2 participants")
            
        # Process each participant
        for entry_idx, participant in enumerate(participant_data):
            logging.info(f"Processing BART NoChat participant {entry_idx + 1}/{len(participant_data)}: {participant.get('participant', 'unknown')}")
            
            try:
                results = process_bart_participant(participant, model, tokenizer, model_key, verbose=test_mode)
                all_results.extend(results)
                    
            except Exception as e:
                logging.error(f"Error processing BART NoChat participant {entry_idx}: {e}")
                continue
        
        # Create and return DataFrame
        if all_results:
            df = pd.DataFrame(all_results)
            logging.info(f"BART NoChat task completed. Generated {len(all_results)} rows of results.")
            
            # Log summary statistics
            if len(all_results) > 0:
                logging.info(f"Choice distribution: {df['human_decision'].value_counts().to_dict()}")
                avg_pumps = df.groupby(['part_id', 'round'])['pumps_so_far'].max().mean()
                logging.info(f"Average pumps per balloon: {avg_pumps:.1f}")
                
                # Log efficiency metrics (similar to original code)
                total_participants = len(participant_data)
                total_decisions = len(all_results)
                logging.info(f"Efficiency: {total_participants} participants processed for {total_decisions} decisions")
                logging.info(f"({total_decisions/total_participants:.0f} decisions per participant)")
            
            return df
        else:
            logging.warning(f"No results generated for BART NoChat task on {model_key}")
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error in BART NoChat task for model {model_key}: {e}")
        return pd.DataFrame()

def get_task_info() -> Dict[str, str]:
    """Return information about this task."""
    return {
        "name": "BART_NoChat",
        "description": "Balloon Analogue Risk Task evaluation using original prefix-based approach (no chat template)",
        "output_columns": ["model", "task", "part_id", "round", "decision_num", "human_decision",
                          "log_prob_pump", "log_prob_stop", "pumps_so_far", "balloon_outcome", 
                          "final_score", "pump_key", "stop_key", "age", "sex", "location"],
        "data_file": DATA_FILE
    }

# For standalone testing
if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
    print("Task info:", get_task_info())