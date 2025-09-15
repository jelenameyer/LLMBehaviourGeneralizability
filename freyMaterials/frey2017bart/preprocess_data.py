import pandas as pd

# Load data
df = pd.read_csv("original_data/bart_pumps.csv", header=0)

# Initiate common format csv
out = pd.DataFrame(columns=['participant', 'partid', 'task', 'trial', 'choice', 'reward'])

# Loop through all trials
for participant, partid in enumerate(df.partid.unique()):

    # Subset of dataset for this participant
    df_part = df[df.partid == partid]

    # Init temporary list for this participant
    temp_list = []

    # Loop through all trials of this participant
    for trial in range(df_part.shape[0]):

        # Loop through pumps in each trial
        for pump in range(df_part.pumps.iloc[trial]):

            # If it is not final pump, add pump
            if pump < df_part.pumps.iloc[trial] - 1:
                temp_list.append([participant, partid, trial, pump, 1, 0])
            # If it is final pump, check if it exploded or not
            else:
                # If it exploded, the balloon was pumped and no reward given
                if df_part.exploded.iloc[trial]:
                    temp_list.append([participant,partid, trial, pump, 1, 0])
                # If the balloon did not explode, the balloon was not pumped and reward extracted
                else:
                    # Add one successful pump to match number of pumps
                    # (if not, we stop one short for non-exploding sequences)
                    temp_list.append([participant, partid, trial, pump, 1, 0])
                    temp_list.append([participant, partid, trial, pump+1, 0, pump+1])

    # Make list to dataframe and concatenate to out dataframe
    out_temp = pd.DataFrame(temp_list, columns=['participant','partid', 'task', 'trial', 'choice', 'reward'])
    out = pd.concat([out, out_temp], ignore_index=True)

# Save file
out.to_csv("exp1.csv")
