"""
CCT (Columbia Card Task) choice prediction scoring script for HuggingFaceTB/SmolLM2-1.7B-Instruct.
This script takes human decision-making data from the CCT task and calculates 
how likely the model finds the final choice decisions given the game history.
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


def parse_cct_data(text: str) -> Dict[str, Any]:
    """
    Parse CCT text and extract all information including game rules and round sequences.
    """
    # Extract game instructions (everything before Round 1)
    round1_start = text.find('Round 1:')
    if round1_start == -1:
        raise ValueError("Could not find Round 1 start")
    
    instructions = text[:round1_start].strip()
    
    # Extract action keys from instructions (randomized per participant)
    turn_match = re.search(r'Press ([A-Z]) to turn a card over', text)
    stop_match = re.search(r'or ([A-Z]) to stop the round', text)
    
    if not turn_match or not stop_match:
        raise ValueError("Could not identify turn and stop keys from instructions")
    
    turn_key = turn_match.group(1)  # Could be V, D, E, etc.
    stop_key = stop_match.group(1)  # Could be L, O, C, etc.
    
    # Split into round sections
    round_pattern = r'Round (\d+):\n(.*?)(?=\nRound \d+:|\Z)'
    round_matches = re.findall(round_pattern, text, re.DOTALL)
    
    rounds = []
    for round_num, round_text in round_matches:
        # Extract round parameters
        gain_match = re.search(r'You will be awarded (\d+) points for turning over a gain card', round_text)
        loss_match = re.search(r'You will lose (\d+) points for turning over a loss card', round_text)
        loss_cards_match = re.search(r'There are (\d+) loss cards in this round', round_text)
        
        gain_points = int(gain_match.group(1)) if gain_match else 0
        loss_points = int(loss_match.group(1)) if loss_match else 0
        loss_cards = int(loss_cards_match.group(1)) if loss_cards_match else 0
        
        # Parse actions within the round
        actions = []
        action_lines = [line for line in round_text.split('\n') if line.strip().startswith('You press')]
        
        current_score = 0
        for line in action_lines:
            if f'<<{turn_key}>>' in line:
                action = 'turn'
                if 'gain card' in line:
                    card_type = 'gain'
                    current_score += gain_points
                elif 'loss card' in line:
                    card_type = 'loss'
                    current_score -= loss_points
                else:
                    card_type = 'unknown'
                
                # Extract current score from line
                score_match = re.search(r'Your current score is (-?\d+)', line)
                if score_match:
                    current_score = int(score_match.group(1))
                    
                actions.append({
                    'action': action,
                    'card_type': card_type,
                    'score_after': current_score
                })
                
            elif f'<<{stop_key}>>' in line:
                action = 'stop'
                actions.append({
                    'action': action,
                    'card_type': None,
                    'score_after': current_score
                })
        
        # Get final score
        final_score_match = re.search(r'Your final score for this round is (-?\d+)', round_text)
        final_score = int(final_score_match.group(1)) if final_score_match else current_score
        
        # Determine if round ended with loss card or stop
        round_ended_loss = 'The round has now ended because you encountered a loss card' in round_text
        
        rounds.append({
            "round_num": int(round_num),
            "gain_points": gain_points,
            "loss_points": loss_points,
            "loss_cards": loss_cards,
            "actions": actions,
            "final_score": final_score,
            "ended_with_loss": round_ended_loss
        })
    
    return {
        "instructions": instructions,
        "turn_key": turn_key,
        "stop_key": stop_key,
        "rounds": rounds
    }


def build_full_sequence_with_decisions(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a single sequence containing all rounds with decision points marked.
    This allows the model to learn from previous rounds.
    """
    instructions = parsed_data["instructions"]
    turn_key = parsed_data["turn_key"]
    stop_key = parsed_data["stop_key"]
    rounds = parsed_data["rounds"]
    
    # Build full text with all rounds
    full_text = instructions
    decision_points = []
    
    for round_data in rounds:
        round_num = round_data["round_num"]
        gain_points = round_data["gain_points"]
        loss_points = round_data["loss_points"]
        loss_cards = round_data["loss_cards"]
        actions = round_data["actions"]
        final_score = round_data["final_score"]
        ended_with_loss = round_data["ended_with_loss"]
        
        # Add round header
        full_text += f"\n\nRound {round_num}:\n"
        full_text += f"You will be awarded {gain_points} points for turning over a gain card.\n"
        full_text += f"You will lose {loss_points} points for turning over a loss card.\n"
        full_text += f"There are {loss_cards} loss cards in this round.\n"
        
        # Track state for decision points
        cards_turned = 0
        current_score = 0
        
        for i, action in enumerate(actions):
            # Mark decision point before each action
            prefix_text = full_text
            if cards_turned > 0:
                # Add previous actions to context
                for prev_action in actions[:i]:
                    if prev_action['action'] == 'turn':
                        prefix_text += f"You press <<{turn_key}>> and turn over a {prev_action['card_type']} card. Your current score is {prev_action['score_after']}."
                        if prev_action['card_type'] == 'loss':
                            prefix_text += " The round has now ended because you encountered a loss card.\n"
                            break
                        else:
                            prefix_text += "\n"
                    elif prev_action['action'] == 'stop':
                        prefix_text += f"You press <<{stop_key}>> and claim your payout.\n"
                        break
            
            prefix_text += "You press <<"
            
            # Store decision information
            decision_points.append({
                "round_num": round_num,
                "decision_num": i + 1,
                "cards_turned": cards_turned,
                "current_score": current_score,
                "choice_made": "turn" if action["action"] == "turn" else "stop",
                "gain_points": gain_points,
                "loss_points": loss_points,
                "loss_cards": loss_cards,
                "round_outcome": "loss" if ended_with_loss else "stop",
                "final_score": final_score,
                "token_position": None  # Will be filled after tokenization
            })
            
            # Update state
            if action["action"] == "turn":
                cards_turned += 1
                current_score = action["score_after"]
        
        # Complete the round text with all actions
        for action in actions:
            if action['action'] == 'turn':
                full_text += f"You press <<{turn_key}>> and turn over a {action['card_type']} card. Your current score is {action['score_after']}."
                if action['card_type'] == 'loss':
                    full_text += " The round has now ended because you encountered a loss card.\n"
                    break
                else:
                    full_text += "\n"
            elif action['action'] == 'stop':
                full_text += f"You press <<{stop_key}>> and claim your payout.\n"
                break
        
        full_text += f"Your final score for this round is {final_score}.\n"
    
    # Tokenize the full sequence once
    device = MODEL.device
    input_ids = TOKENIZER(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    
    # Find token positions for each decision point
    updated_decision_points = []
    
    for decision in decision_points:
        round_num = decision["round_num"]
        decision_num = decision["decision_num"]
        
        # Rebuild prefix up to this decision
        prefix = instructions
        
        # Add all previous rounds
        for prev_round in rounds:
            if prev_round["round_num"] >= round_num:
                break
            
            prefix += f"\n\nRound {prev_round['round_num']}:\n"
            prefix += f"You will be awarded {prev_round['gain_points']} points for turning over a gain card.\n"
            prefix += f"You will lose {prev_round['loss_points']} points for turning over a loss card.\n"
            prefix += f"There are {prev_round['loss_cards']} loss cards in this round.\n"
            
            # Add all actions from previous round
            for action in prev_round['actions']:
                if action['action'] == 'turn':
                    prefix += f"You press <<{turn_key}>> and turn over a {action['card_type']} card. Your current score is {action['score_after']}."
                    if action['card_type'] == 'loss':
                        prefix += " The round has now ended because you encountered a loss card.\n"
                        break
                    else:
                        prefix += "\n"
                elif action['action'] == 'stop':
                    prefix += f"You press <<{stop_key}>> and claim your payout.\n"
                    break
            
            prefix += f"Your final score for this round is {prev_round['final_score']}.\n"
        
        # Add current round up to this decision
        current_round = rounds[round_num - 1]  # 0-indexed
        prefix += f"\n\nRound {round_num}:\n"
        prefix += f"You will be awarded {current_round['gain_points']} points for turning over a gain card.\n"
        prefix += f"You will lose {current_round['loss_points']} points for turning over a loss card.\n"
        prefix += f"There are {current_round['loss_cards']} loss cards in this round.\n"
        
        # Add previous actions in this round
        actions_before = current_round['actions'][:decision_num - 1]
        for action in actions_before:
            if action['action'] == 'turn':
                prefix += f"You press <<{turn_key}>> and turn over a {action['card_type']} card. Your current score is {action['score_after']}."
                if action['card_type'] == 'loss':
                    prefix += " The round has now ended because you encountered a loss card.\n"
                    break
                else:
                    prefix += "\n"
            elif action['action'] == 'stop':
                prefix += f"You press <<{stop_key}>> and claim your payout.\n"
                break
        
        prefix += "You press <<"
        
        # Tokenize prefix to find position
        prefix_tokens = TOKENIZER(prefix, add_special_tokens=False).input_ids
        decision["token_position"] = len(prefix_tokens)
        
        updated_decision_points.append(decision)
    
    return {
        "input_ids": input_ids,
        "decision_points": updated_decision_points,
        "turn_key": turn_key,
        "stop_key": stop_key
    }


def get_all_decision_logprobs(sequence_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get log probabilities for all decisions in one forward pass using masking.
    """
    input_ids = sequence_data["input_ids"]
    decision_points = sequence_data["decision_points"]
    turn_key = sequence_data["turn_key"]
    stop_key = sequence_data["stop_key"]
    
    # Single forward pass for the entire sequence
    with torch.inference_mode():
        outputs = MODEL(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)
    
    # Get token IDs for turn and stop keys
    turn_token = TOKENIZER(turn_key, add_special_tokens=False).input_ids[0]
    stop_token = TOKENIZER(stop_key, add_special_tokens=False).input_ids[0]
    
    results = []
    
    for decision in decision_points:
        pos = decision["token_position"]
        
        # Get logits at the decision position (predict next token after "<<")
        pred_logits = logits[0, pos - 1, :]  # -1 because we predict the next token
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        
        result = {
            "round_num": decision["round_num"],
            "decision_num": decision["decision_num"],
            "cards_turned": decision["cards_turned"],
            "current_score": decision["current_score"],
            "choice_made": decision["choice_made"],
            "gain_points": decision["gain_points"],
            "loss_points": decision["loss_points"],
            "loss_cards": decision["loss_cards"],
            "round_outcome": decision["round_outcome"],
            "final_score": decision["final_score"],
            "log_prob_turn": log_probs[turn_token].item(),
            "log_prob_stop": log_probs[stop_token].item(),
            "turn_key": turn_key,
            "stop_key": stop_key,
        }
        
        results.append(result)
    
    return results


def process_cct_participant(participant_data: dict, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Process a single CCT participant using efficient sequence processing.
    """
    text = participant_data["text"]
    participant_id = participant_data["participant"]
    
    try:
        # Parse the data
        parsed_data = parse_cct_data(text)
        
        # Build sequence with decision points
        sequence_data = build_full_sequence_with_decisions(parsed_data)
        
        # Get predictions for all decisions in one pass
        decision_results = get_all_decision_logprobs(sequence_data)
        
        # Add participant metadata
        for result in decision_results:
            result.update({
                "participant_id": participant_id,
                "model": MODEL_KEY,
                "experiment": participant_data.get("experiment"),
            })
        
        if verbose:
            total_decisions = len(decision_results)
            total_rounds = len(parsed_data['rounds'])
            print(f"Participant {participant_id}: {total_decisions} decisions across {total_rounds} rounds")
        
        return decision_results
        
    except Exception as e:
        logging.error(f"Error processing participant {participant_id}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Score CCT decisions with SmolLM2-1.7B-Instruct using efficient masking.")
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
    output_path = os.path.join(OUTPUTS_DIR, f"{MODEL_KEY}_cct_results.csv")

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
            results = process_cct_participant(participant, verbose=args.verbose)
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
            
            print(f"\nRound outcome distribution:")
            print(results_df.groupby(['participant_id', 'round_num'])['round_outcome'].first().value_counts())
            
            avg_cards_per_round = results_df.groupby(['participant_id', 'round_num'])['cards_turned'].max().mean()
            print(f"\nAverage cards turned per round: {avg_cards_per_round:.1f}")
            
            # Show model efficiency
            total_forward_passes = len(participant_data)  # One per participant
            total_decisions = len(all_results)
            print(f"\nEfficiency: {total_forward_passes} forward passes for {total_decisions} decisions")
            print(f"({total_decisions/total_forward_passes:.0f} decisions per forward pass)")
    else:
        logging.warning("No results to save.")

if __name__ == "__main__":
    main()