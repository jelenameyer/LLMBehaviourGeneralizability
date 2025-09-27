import pandas as pd
import os
import json
import numpy as np

base_path = ""
folders = [""]

output_prompts = []


# helper functions 
# randomizes letters that are used to generate prompts (because of prompt sensitivity in LLMs)
def randomized_choice_options(num_choices):
    choice_options = list(map(chr, range(65, 91)))  # A–Z
    return np.random.choice(choice_options, num_choices, replace=False)


def format_number(num):
    return f"{int(num)}" if num == int(num) else f"{num:.1f}"


def convert_to_builtin_type(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# dictionary that maps gamble labels to descriptions
gamble_descriptions = {
    "he04_1": "Box A: 4 points (80%) or 0 points (20%) vs. Box B: 3 points (100%)",
    "he04_2": "Box A: 4 points (20%) or 0 points (80%) vs. Box B: 3 points (25%) or 0 points (75%)",
    "he04_3": "Box A: -3 points (100%) or 0 points (0%) vs. Box B: -32 points (10%) or 0 points (90%)",
    "he04_4": "Box A: -3 points (100%) or 0 points (0%) vs. Box B: -4 points (80%) or 0 points (20%)",
    "he04_5": "Box A: 32 points (10%) or 0 points (90%) vs. Box B: 3 points (100%)",
    "he04_6": "Box A: 32 points (2.5%) or 0 points (97.5%) vs. Box B: 3 points (25%) or 0 points (75%)",
    "he04_2inv": "Box A: -3 points (25%) or 0 points (75%) vs. Box B: -4 points (20%) or 0 points (80%)",
    "he04_6inv": "Box A: -3 points (25%) or 0 points (75%) vs. Box B: -32 points (2.5%) or 0 points (97.5%)",
}


for folder in folders:
    dfd_samples_path = os.path.join(base_path, folder, "orig_data_dfd", "dfd_perprob.csv")
    print(dfd_samples_path)
    dfd_samples = pd.read_csv(dfd_samples_path)
    print(dfd_samples.head())
    participants_path = os.path.join(base_path, folder, "orig_data_dfd", "participants.csv")
    print(participants_path)
    participants = pd.read_csv(participants_path)
    print(participants.head())

    for participant_id in dfd_samples["partid"].unique():
        participant_data = dfd_samples[dfd_samples["partid"] == participant_id]
        meta_row = participants[participants["partid"] == participant_id]

        # assign random labels ONCE per participant
        choice_labels = randomized_choice_options(2)
        label_A, label_B = choice_labels[0], choice_labels[1]

        trials_text = []
        all_rts = []

        session_text = (
            f"In each round, you will be presented with two boxes: Box {label_A} and Box {label_B}. "
            "Each box contains different possible point amounts, with some amounts being more frequent than others. "
            "These points are later converted into real money. \n"
            "You will receive a written description of the possible outcomes in each box, along with the probabilities of these outcomes.\n"
            "Your task is to carefully consider these descriptions and then choose which box you prefer.\n"
            "Once you make your choice, one outcome will be randomly drawn from the box you selected. That outcome will determine how many points you earn or lose for that round.\n"
            "At the end of the study, one task will be randomly selected. The outcome from that task will be added to or subtracted from your starting bonus of 15 CHF or 10 EUR. "
            "Depending on your decisions, you could double your bonus or lose it entirely."
        )

        for trial_number, trial_data in participant_data.groupby("gamble_ind"):
            trial_text = [f"Problem {int(trial_number)}: Please choose one box by typing the corresponding letter."]

            # look up gamble description
            gamble_lab = trial_data.iloc[0]["gamble_lab"]
            description = gamble_descriptions.get(gamble_lab, "Description not found")

            # replace A/B with randomized labels
            description = description.replace("Box A", f"Box {label_A}").replace("Box B", f"Box {label_B}")
            trial_text.append(description)

            # get final decision
            final_decision = trial_data.iloc[-1]
            if final_decision.decision == "A":
                final_choice_label = label_A
            else:
                final_choice_label = label_B

            trial_text.append(
                f"You choose Box <<{final_choice_label}>> based on the description."
            )

            trials_text.append("\n".join(trial_text))

            # save RT
            all_rts.append(int(final_decision.rt_decision) if not pd.isna(final_decision.rt_decision) else None)

        participant_text = session_text + "\n\n" + "\n\n".join(trials_text)

        # meta info
        meta_info = {}
        if not meta_row.empty:
            for field in ["sex", "age", "location"]:
                if field in meta_row.columns and not pd.isnull(meta_row.iloc[0][field]):
                    meta_info[field] = meta_row.iloc[0][field]

        prompt = {
            "participant": f"{participant_id}",
            "experiment": "Decisions From Description",
            "text": participant_text,
            "RTs": all_rts,
        }
        #print(participant_text)

        if meta_info:
            prompt.update(meta_info)

        output_prompts.append(prompt)

# save results
output_path = "prompts_dfd.jsonl"
with open(output_path, "w") as f:
    for prompt in output_prompts:
        prompt = {k: convert_to_builtin_type(v) for k, v in prompt.items()}
        f.write(json.dumps(prompt, ensure_ascii=False) + "\n")
