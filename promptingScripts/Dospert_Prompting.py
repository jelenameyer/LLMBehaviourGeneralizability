#!/usr/bin/env python3
"""
DOSPERT Task Module - Standalone task for DOSPERT evaluation.
This module can be imported and run by the model manager.
"""

import json
import torch
import pandas as pd
import re
import logging
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# Task-specific configuration
DATA_FILE = "survey_data/promptsDospertVaried.jsonl"

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
        #print(f"Tokens used: {user_tok, assist_tok}")
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

def candidate_logprobs_chatstyle(text: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_key: str) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts with logprobs for candidates 1–5
    at each << >> position, using chat template formatting.
    All models are instruction-tuned so they all use chat templates.
    """
    # Use chat template approach (all models have them)
    USER_TOK, ASSIST_TOK = detect_chat_tokens(tokenizer)
    
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
    #print(chat_text)
    
    # Encode and get logprobs
    enc = tokenizer(chat_text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = enc.input_ids.to(model.device)
    offsets = enc.offset_mapping[0].tolist()
    
    # Find assistant responses
    pattern = re.compile(rf"{re.escape(ASSIST_TOK)}\s*(\d)")
    search_text = chat_text

    # Compute logprobs
    with torch.no_grad():
        out = model(input_ids)
        logprobs = torch.nn.functional.log_softmax(out.logits, dim=-1)[0]

    # Candidate IDs 1–5
    try:
        cand_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]
    except IndexError:
        # Fallback if tokenizer doesn't have single-token numbers
        logging.warning("Could not get single-token IDs for 1-5, using multi-token encoding")
        cand_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 6)]
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

def run_task(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, model_key: str, 
             test_mode: bool = False, data_file: str = DATA_FILE) -> pd.DataFrame:
    """
    Main task runner function. This is the interface that the model manager will call.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        model_key: String identifier for the model
        test_mode: Whether to run in test mode (fewer entries)
        data_file: Path to the data file
    
    Returns:
        pandas.DataFrame: Results dataframe with columns for logprobs
    """
    logging.info(f"Starting DOSPERT task for model: {model_key}")
    
    all_rows = []
    
    try:
        # Load data
        with open(data_file) as f:
            entries = [json.loads(line) for line in f]
            
        # Limit entries in test mode
        if test_mode:
            entries = entries[:2]
            logging.info(f"Test mode: processing only first 2 entries")
            
        # Process each entry
        for entry_idx, entry in enumerate(entries):
            logging.info(f"Processing DOSPERT entry {entry_idx + 1}/{len(entries)}")
            
            try:
                spans = candidate_logprobs_chatstyle(entry["text"], model, tokenizer, model_key)
                
                for i, s in enumerate(spans, 1):
                    s["model"] = model_key
                    s["item"] = i
                    s["participant"] = entry["participant"]
                    s["flipped"] = entry.get("flipped", "")
                    s["experiment"] = entry.get("experiment", "")
                    all_rows.append(s)
                    
            except Exception as e:
                logging.error(f"Error processing DOSPERT entry {entry_idx}: {e}")
                continue
        
        # Create and return DataFrame
        if all_rows:
            df = pd.DataFrame(all_rows)
            logging.info(f"DOSPERT task completed. Generated {len(all_rows)} rows of results.")
            return df
        else:
            logging.warning(f"No results generated for DOSPERT task on {model_key}")
            return pd.DataFrame()  # Return empty DataFrame
            
    except Exception as e:
        logging.error(f"Error in DOSPERT task for model {model_key}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def get_task_info() -> Dict[str, str]:
    """Return information about this task."""
    return {
        "name": "DOSPERT",
        "description": "DOSPERT risk-taking questionnaire evaluation using logprobs",
        "output_columns": ["model", "item", "participant", "flipped", "experiment", 
                          "human_number", "1", "2", "3", "4", "5"],
        "data_file": DATA_FILE
    }

# For standalone testing
if __name__ == "__main__":
    print("This is a task module meant to be imported by model_manager.py")
    print("Task info:", get_task_info())