import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import re

# ---- CONFIG ----
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE = "survey_data/subsample_prompts_DospertVaried.jsonl"

# ---- Load model ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Chat template: {tokenizer.chat_template}")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model.to(DEVICE).eval()
print(f"Loaded model: {MODEL_NAME}")

# ---- Auto-detect chat tokens ----
def detect_chat_tokens(tokenizer):
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

USER_TOK, ASSIST_TOK = detect_chat_tokens(tokenizer)
print(f"Using USER_TOK={USER_TOK} ASSIST_TOK={ASSIST_TOK}")


# ---- Candidate logprobs function ----
def candidate_logprobs_chatstyle(text):
    """
    Returns a list of dicts with logprobs for candidates 1–5
    at each << >> position, but with user/assistant markers inserted.
    """
    # rebuild text into chat form
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
    print(chat_text)

    # Encode text and get offsets
    enc = tokenizer(chat_text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = enc.input_ids.to(DEVICE)
    offsets = enc.offset_mapping[0].tolist()

    # Compute logprobs
    with torch.no_grad():
        out = model(input_ids)
        logprobs = torch.nn.functional.log_softmax(out.logits, dim=-1)[0]

    # Candidate IDs 1–5
    cand_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]

    results = []
    # Regex: match assistant token followed by the numeric answer
    pattern = re.compile(rf"{re.escape(ASSIST_TOK)}\s*(\d)")
    for m in pattern.finditer(chat_text):
        human = m.group(1)
        span_lo, span_hi = m.span(1)
        # Find token index overlapping with the number
        tok_idx = next(
            i for i, (lo, hi) in enumerate(offsets)
            if not (hi <= span_lo or lo >= span_hi)
        )
        # Extract logprobs for candidates 1–5 at this position
        lp_candidates = {str(k): logprobs[tok_idx][cid].item()
                         for k, cid in zip(range(1, 6), cand_ids)}
        results.append(dict(human_number=human, **lp_candidates))

    return results

# ---- Main ----
all_rows = []

with open(DATA_FILE) as f:
    for line in f:
        entry = json.loads(line)
        spans = candidate_logprobs_chatstyle(entry["text"])
        for i, s in enumerate(spans, 1):
            s["item"] = i
            s["participant"] = entry["participant"]
            s["flipped"] = entry.get("flipped", "")
            s["experiment"] = entry.get("experiment", "")
            all_rows.append(s)

pd.DataFrame(all_rows).to_csv("dospert_llm_scores_chatstyle.csv", index=False)
print("Saved dospert_llm_scores_chatstyle.csv")




# --------------------------------------------------------------------------------------------------------------
# import json
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from pathlib import Path
# import pandas as pd
# import re

# # ---- CONFIG ----
# MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# MODEL_KEY = "SmolLM2-1.7B-Instruct"  
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DATA_FILE = "survey_data/subsample_prompts_DospertVaried.jsonl"  # each line = {"participant":..., "text":...}

# # ---- Load model ----
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
# model.to(DEVICE)
# model.eval()
# print(f"loaded model: {MODEL_KEY}")


# def candidate_logprobs(text):
#     """
#     Returns a list of dicts with logprobs for candidates 1–5 at each << >> position.
#     """
#     enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
#     input_ids  = enc.input_ids.to(DEVICE)          # [1, seq_len]
#     offsets    = enc.offset_mapping[0].tolist()    # char positions of each token
#     with torch.no_grad():
#         out = model(input_ids)
#         # log p(x_t | x_<t) predicts token t from logits at t-1
#         logprobs = torch.nn.functional.log_softmax(out.logits, dim=-1)[0]  # [seq_len, vocab]

#     # candidate token IDs for "1"..."5"
#     cand_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1,6)]

#     results = []
#     # find every 'N. ... <<d>>'
#     pattern = re.compile(r"(\d+)\..*?<<(.)>>")
#     for m in pattern.finditer(text):
#         item   = int(m.group(1))
#         human  = m.group(2)
#         # index of the number token in input_ids
#         span_lo, span_hi = m.span(2)
#         # token position whose prediction generates this digit
#         tok_idx = next(
#             i for i,(lo,hi) in enumerate(offsets)
#             if not (hi<=span_lo or lo>=span_hi)
#         )

#         # model predicts token[tok_idx] at position tok_idx
#         lp_candidates = {str(k): logprobs[tok_idx][cid].item()
#                          for k,cid in zip(range(1,6), cand_ids)}

#         results.append(dict(item=item,
#                             human_number=human,
#                             **lp_candidates))
#     return results


# all_rows = []
# with open(DATA_FILE) as f:
#     for line in f:
#         entry = json.loads(line)
#         text  = entry["text"]
#         spans = candidate_logprobs(text) 
#         print(spans)
#         for s in spans:
#             s["participant"] = entry["participant"]
#             s["flipped"]     = entry["flipped"]
#             s["experiment"] = entry["experiment"]
#             all_rows.append(s)

# pd.DataFrame(all_rows).to_csv("dospert_llm_scores_all_candidates.csv", index=False)
# print("Saved dospert_llm_scores_all_candidates.csv")

