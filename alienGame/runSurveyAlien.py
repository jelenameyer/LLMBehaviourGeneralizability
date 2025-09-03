# ---------------------- import important packages --------------------------------------------------
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines
import pandas as pd
import random
#print(sys.orig_argv)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disabling parallelism, because I need to loop over different models one after another for now!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Helper function to send prompt to LLM and get log-probs and answer back -----------------------------------------

def query_llm(prompt, max_new_tokens=13, temperature=0.2, top_p=0.9):
    """Send a prompt to the model, return answer and log-probabilities of first new token."""
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Forward pass to get logits
        outputs = model(**inputs)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]

        # Log-probs for the next token
        next_token_logits = logits[0, -1, :]
        log_probs = F.log_softmax(next_token_logits, dim=-1)

        # Generate continuation
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    # Extract generated answer
    new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Convert log-probs tensor to Python dict (top-10 only, to keep things small)
    topk = torch.topk(log_probs, 10)
    log_probs_dict = {
        tokenizer.decode([idx.item()]): score.item()
        for idx, score in zip(topk.indices, topk.values)
    }

    return answer, log_probs_dict


# ---------------------- Load models & tokenizer -----------------------------------------


# some parameters
records = []
num_trials = 5

model_names = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]#, "Qwen/Qwen3-8B"]

for model_name in model_names: 
    last_answers = {}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)


    # Example "experiment" loop with several rounds/trials 
    instructions = "You are playing the Alien Game. \n There are 10 symbols. Each symbol can be selected (1) or deselected (0). \n The alien has hidden preferences and will pay a certain amount for each picture you submit. \n Submit your picture as a 10-digit binary string. \n After each submission, you will be told the payoff."
    
    for trial in range(1, num_trials + 1):
        payoff = random.randint(1, 20)
        # Build history string from all previous trials
        history_str = ""
        for past_trial in range(1, trial):
            prev = last_answers[past_trial]
            history_str += (
                f"\nTrial {past_trial}:\n"
                f"  Submitted: {prev['answer']}\n"
                f"  Payoff: {prev['payoff']}\n"
            )

        prompt = (
            instructions
            + (f"\n\nGame history so far:{history_str}" if history_str else "")
            + f"\n\nTrial {trial}:  \nSymbols: 1 2 3 4 5 6 7 8 9 10 \n"
            "Your choice? \nPlease ONLY provide your choice, NO REASONING!"
        )

        answer, log_probs = query_llm(prompt)

        # Save answer + payoff for this trial
        last_answers[trial] = {
            "answer": answer,
            "payoff": payoff,
        }

        records.append({
            "model_name": model_name,
            "trial": trial,
            "prompt": prompt,
            "answer": answer,
            "log_probs": log_probs,
        })

    print(f"Model **{model_name}** done!")

# ---------------------- Save results ---------------------------------------------------
# To pandas DataFrame
df = pd.DataFrame(records)
df.to_csv("llm_playthrough.csv", index=False)

# To JSONL
with jsonlines.open("llm_playthrough.jsonl", mode="w") as writer:
    writer.write_all(records)










# # ---------------------- Saving old versions -----------------------------------------


# # ---------------------- Load model & tokenizer -----------------------------------------
# model_name = "Qwen/Qwen3-1.7B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model.eval()
# model.to(device)

# # ---------------------- Helper function -----------------------------------------

# def query_llm(prompt, max_new_tokens=3, temperature=0.2, top_p=0.9):
#     """Send a prompt to the model, return answer and log-probabilities of first new token."""
#     # Prepare input
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         # Forward pass to get logits
#         outputs = model(**inputs)
#         logits = outputs.logits  # shape: [1, seq_len, vocab_size]

#         # Log-probs for the next token
#         next_token_logits = logits[0, -1, :]
#         log_probs = F.log_softmax(next_token_logits, dim=-1)

#         # Generate continuation
#         generated = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=temperature,
#             top_p=top_p,
#         )

#     # Extract generated answer
#     new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
#     answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

#     # Convert log-probs tensor to Python dict (top-10 only, to keep things small)
#     topk = torch.topk(log_probs, 10)
#     log_probs_dict = {
#         tokenizer.decode([idx.item()]): score.item()
#         for idx, score in zip(topk.indices, topk.values)
#     }

#     return answer, log_probs_dict

# # ---------------------- Example "experiment" loop --------------------------------------
# instructions = "Please answer with an integer, either 0 or 1. Choose 0 or 1 and nothing else."

# num_trials = 5
# records = []

# for trial in range(1, num_trials + 1):
#     prompt = instructions + f"\nTrial {trial}: Please choose an integer, 0 or 1."
#     answer, log_probs = query_llm(prompt)

#     records.append({
#         "model_name": model_name,
#         "trial": trial,
#         "prompt": prompt,
#         "answer": answer,
#         "log_probs": log_probs,
#     })


# OLDEST

# model_name = "Qwen/Qwen3-0.6B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model.eval()

# # Chat messages
# messages = [
#     {"role": "user", "content": "What is the capital of France?"},
# ]

# # ---------------------- Prepare input ---------------------------------------------------
# inputs = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     enable_thinking=False,  
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(model.device)

# with torch.no_grad():
#     outputs = model(**inputs)
#     logits = outputs.logits  # shape: [1, seq_len, vocab_size]

# # Compute log-probabilities for the next token
# next_token_logits = logits[0, -1, :]
# log_probs = F.log_softmax(next_token_logits, dim=-1)

# # Show top-10 most probable tokens
# print("Top-10 next tokens:")
# topk = torch.topk(log_probs, 10)
# for idx, score in zip(topk.indices, topk.values):
#     print(f"{tokenizer.decode([idx.item()])!r} -> {score.item():.4f}")

# # -------------------- Generate full answer from Qwen -------------------------------------
# with torch.no_grad():
#     generated_ids = model.generate(
#         **inputs,
#         max_new_tokens=50
#     )

# answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

# print("\nQwen’s answer:")
# print(answer)

