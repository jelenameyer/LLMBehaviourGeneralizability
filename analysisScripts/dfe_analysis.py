# ----------- task description -----------------------------------------
# this script reads the full dfe data and produces a subset where decision (A/B) per round (1-8) for every person are collected.
# next, it reads the LLM responses, normalizes their logprobs per possible answer item and merges the human and llm responses together.

# I then multiply the probablities per round of the really given answer with each other and get one total probabiity the LLM assigns per person. 

# ------------------- load packages -------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ------------- load all data --------------------------
# read human data
full_trial_data_humans = pd.read_csv("data/dfe_samples.csv")

# read the LLM generated data with logprobs 
# read the jsonl file
llm_responses = pd.read_csv("data/SmolLM2-1.7B-Instruct_dfe_results.csv")


 


# --------------- preprocess both dfs -----------------------

# ----------- human data --------------------
# subset human data, leave only one row per round per person (i.e. 8 per person)
subset_per_trial_humans = (
    full_trial_data_humans
    .groupby(["partid", "gamble_ind"], as_index=False)
    .first()
)
 # subset important columns to keep
subset_per_trial_humans = subset_per_trial_humans[["partid", "gamble_lab", "gamble_ind", "decision"]]

# if needed, save human data
#subset_per_trial_humans.to_csv("data/dfe_per_round_data.csv", index=False)


# ----------- LLM data -----------------------

# extract A and B logprobs as numpy array of shape (n_rows, 2)
logprobs = llm_responses[["log_prob_A", "log_prob_B"]].to_numpy()

# apply softmax row-wise for numerical stability
exp_shifted = np.exp(logprobs - logprobs.max(axis=1, keepdims=True))
probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

# make a DataFrame with the normalized probabilities
probs_df = pd.DataFrame(probs, columns=["prob_A", "prob_B"], index=llm_responses.index)

# add back to df
llm_responses_and_probs = pd.concat([llm_responses, probs_df], axis=1)

# keep only the columns I want
subset_llms = llm_responses_and_probs[["participant_id", "model", "round", "prob_A", "prob_B"]]
subset_llms["experiment"] = "DFE"

# -------------------- sanity check -----------------------------------

# check later with full data:
print(len(subset_llms) == len(subset_per_trial_humans))
print(len(subset_llms))
print(len(subset_per_trial_humans))


# ------------------- merge original human data with LLM answers -------------

merged = pd.merge(
    subset_per_trial_humans,
    subset_llms,
    left_on=["partid", "gamble_ind"],
    right_on=["participant_id", "round"],
    how="inner"   # to only have data where both have data 
)

# keep only the columns I want
merged = merged[["experiment", "partid", "gamble_lab", "gamble_ind", "decision", "prob_A", "prob_B"]]


# save merged df to csv
merged.to_csv("data/human_and_llm_responses.csv", index=False)

print(len(merged))


rows_with_na = merged[merged.isna().any(axis=1)]
#print(rows_with_na)
# two participants only played 7 rounds (65020301, 65062001), i.e. no data for round 8
# but for the same participants when merged on right there are there probablities for two rows without participant info too. 
# excluded these two rows for now!

# --------------------- calc total probability per person -----------------------

# Step 1: Compute probability of actual choice for each row
merged['prob_actual'] = merged.apply(lambda row: row['prob_A'] if row['decision'] == 'A' else row['prob_B'], axis=1)

# Step 2: Compute joint probability per participant (over all 8 rounds)
participant_probs = merged.groupby('partid')['prob_actual'].agg(lambda x: np.exp(np.sum(np.log(x))))

# Step 3: Add as a new column to the original dataframe
merged_with_probs = merged.merge(participant_probs.rename('overall_prob'), on='partid')


# ------------------ add the DV of the humans and calc final DV outcome for LLM (through weighting with probabilities) ------------

# load original data with DF = DFEre = Rexp
outcome_data_humans = pd.read_csv("data/dfe_perpers.csv")

data_to_analyze = pd.merge(
    merged_with_probs,
    outcome_data_humans,
    on = "partid",
    how="left"   # to save all rows from the merged data
)

data_to_analyze = data_to_analyze[["experiment", "partid", "gamble_lab", "gamble_ind", "decision", "prob_A", "prob_B", "overall_prob", "Rexp"]]


# save merged df to csv
data_to_analyze.to_csv("data/human_and_llm_responses.csv", index=False)
