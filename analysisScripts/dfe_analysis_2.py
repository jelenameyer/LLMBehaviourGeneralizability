
# ------------------- load packages -------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read data
data_to_analyze = pd.read_csv("data/human_and_llm_responses.csv")


#vis_data = data_to_analyze.groupby("partid", as_index=False)[["overall_prob", "Rexp", "partid"]].first()
vis_data = data_to_analyze[data_to_analyze["gamble_lab"] == "he04_6inv"]
print(vis_data["Rexp"].value_counts())
print(len(vis_data["Rexp"].value_counts()))


# most probable and unprobable LLM answers
# Top 10 most likely
top10 = vis_data.nlargest(10, "overall_prob")[["Rexp", "overall_prob"]]

# Top 10 least likely
bottom10 = vis_data.nsmallest(10, "overall_prob")[["Rexp", "overall_prob"]]

# Combine them into one table (optional)
result = pd.concat(
    [top10.assign(type="most likely"), bottom10.assign(type="least likely")]
)

print(result)
# ---------------- visualise ----------------------------

#sns.countplot(data=data_to_analyze, x='overall_prob')

# plot with seaborn
# sns.barplot(x="Rexp", y="overall_prob", data=vis_data)
# plt.title("Distribution LLM probability estimates of human DFEre DV values")
# plt.xlabel("DFEre")
# plt.ylabel("Probability assigned by LLM")
# plt.show()


plt.scatter(x= vis_data["Rexp"], y = vis_data["overall_prob"])
plt.xlabel("DV")
plt.ylabel("LLM assigned Probability")
plt.title("Distribution of LLM probabilities for different human DV outcomes")
plt.show()
