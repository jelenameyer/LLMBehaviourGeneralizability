# ----------- task description -----------------------------------------
#
# read human pump BART data and LLM responses, calculate per participant normalized probabibilities, merge and save human and LLM data per participant
#
# ------------------- load packages -------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ------------- load all data --------------------------
# read human data, necessary?
human_data = pd.read_csv("bart_data/bart.csv")

# read the LLM generated data with logprobs 
llm_responses = pd.read_csv("bart_data/SmolLM2-1.7B-Instruct_bart_results.csv")

# print(llm_responses.head())
# print(human_data.head())


# --------------- preprocess dfs -----------------------

# ----------- LLM data -----------------------

# extract logprobs as numpy array of shape (n_rows, 2)
logprobs = llm_responses[["log_prob_pump", "log_prob_stop"]].to_numpy()

# apply softmax row-wise for numerical stability
exp_shifted = np.exp(logprobs - logprobs.max(axis=1, keepdims=True))
probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

# make a DataFrame with the normalized probabilities
probs_df = pd.DataFrame(probs, columns=["prob_pump", "prob_stop"], index=llm_responses.index)

# add back to df
llm_responses_and_probs = pd.concat([llm_responses, probs_df], axis=1)

# keep only the columns I want
llm_responses_and_probs["experiment"] = "BART"


# --------------------- calc total probability per person -----------------------

# Step 1: Compute probability of actual choice for each row
llm_responses_and_probs['prob_actual'] = llm_responses_and_probs.apply(lambda row: row['prob_pump'] if row['choice_made'] == 'pump' else row['prob_stop'], axis=1)

# Step 2: compute average log probability (per trial) -> geometric mean
participant_probs = llm_responses_and_probs.groupby('participant_id')['prob_actual'].agg(
    lambda x: np.exp(np.mean(np.log(x)))
)

# Step 3: merge back
llm_responses_and_probs = llm_responses_and_probs.merge(
    participant_probs.rename('normalized_prob'),
    on='participant_id'
)

# print(llm_responses_and_probs.head())



# ------------------- merge original human data with LLM answers -------------

# therefore we only need the participant id and assigned LLM probabiity per participant
subset = llm_responses_and_probs[["participant_id", "normalized_prob", "experiment"]].drop_duplicates()


# ------- sanity check ---------

# check later with full data:
print(len(subset) == len(human_data))
print(len(subset))
print(len(human_data))



merged = pd.merge(
    human_data,
    subset,
    left_on=["partid"],
    right_on=["participant_id"],
    how="inner"   # to only have data where both have data 
)

# # keep only the columns I want
merged = merged[["experiment", "partid", "normalized_prob", "pumps", "pumps_adj"]]
print(merged.head())

# save merged df to csv
merged.to_csv("bart_data/human_and_llm_responses_bart.csv", index=False)







# ----------- some analyses ----------------------

# most probable and unprobable LLM answers
# Top 10 most likely
top10 = merged.nlargest(10, "normalized_prob")[["pumps", "normalized_prob"]]

# Top 10 least likely
bottom10 = merged.nsmallest(10, "normalized_prob")[["pumps", "normalized_prob"]]

# Combine them into one table (optional)
result = pd.concat(
    [top10.assign(type="most likely"), bottom10.assign(type="least likely")]
)

print(result)
# ---------------- visualise ----------------------------

plt.scatter(x= merged["pumps"], y = merged["normalized_prob"])
plt.xlabel("DV")
plt.ylabel("LLM assigned Normalized Probability")
plt.title("Distribution of LLM probabilities for different human DV outcomes")
plt.show()
