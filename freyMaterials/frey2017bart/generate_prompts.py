import numpy as np
import pandas as pd
import jsonlines
import sys
sys.path.append("..")

# helper function (originally in uils.py (Dirk sent me isolated function)
# randomizes letters that are used to generate prompts (because of prompt sensitivity LLMs)
def randomized_choice_options(num_choices):
    choice_options = list(map(chr, range(65, 91)))
    return np.random.choice(choice_options, num_choices, replace=False)


datasets = ["exp1.csv"]
all_prompts = []

for dataset in datasets:
    df = pd.read_csv(dataset)
    num_participants = df.participant.max() + 1
    

    for participant in range(num_participants):
        df_participant = df[(df['participant'] == participant)]
        choice_options = randomized_choice_options(num_choices=2)

        # get current partid int (so that we have the number to compare intrapersonal correlations later)
        partid_value = df_participant['partid'].iloc[0]
        #print(partid_value)

        prompt = 'Throughout the task, you will be presented with balloons, one at a time.\n'\
                 'In each step, you can choose to pump up the balloon by pressing ' + choice_options[1] + ' and you will accumulate 1 point for each pump.\n'\
                 'At any point, you can stop pumping up the balloon by pressing ' + choice_options[0] + ' and you will collect your accumulated points.\n'\
                 'You will repeat this procedure on multiple different balloons.\n'\
                 'It is your choice to determine how much to pump up the balloon, but be aware that at some point the balloon will explode.\n'\
                 'If the balloon explodes before you collect your accumulated points, then you move on to the next balloon and the points are lost.\n\n'

        num_tasks = df_participant.task.max() + 1
        for task in range(num_tasks):
            df_task = df_participant[(df_participant['task'] == task)]
            prompt += 'Balloon ' + str(task + 1) + ':\nYou press'

            # Number of "steps" (trials) is always different for each "task" (block)
            for trial in df_task.trial:
                df_trial = df_task[(df_task['trial'] == trial)]
                c = df_trial.choice.item()
                r = df_trial.reward.item()
                prompt += ' {'+str(choice_options[c])+'}'

            # Add final sentence
            if r == 0:
                #prompt = prompt[:-1]
                prompt += '. The balloon was inflated too much and explodes.\n\n'
            else:
                #prompt = prompt[:-1]
                prompt += '. You stop inflating the balloon and get ' + str(r) + ' points.\n\n'

        # Print and append prompt
        prompt = prompt[:-2]
        #print(prompt)
        all_prompts.append({'participant': str(partid_value),
                            'experiment': 'frey2017risk/BART.csv', 
                            'text': prompt})

# Write all to output file
with jsonlines.open('prompts_bart.jsonl', 'w') as writer:
    writer.write_all(all_prompts)
