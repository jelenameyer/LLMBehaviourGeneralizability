import pandas as pd
import random
import jsonlines
import sys
sys.path.append("..")
from utils import randomized_choice_options

all_prompts = []
df = pd.read_csv("exp1.csv", dtype='Int32')
print(len(df.participant.unique()))

for participant in df.participant.unique():
    print(participant)
    df_participant = df[(df['participant'] == participant)]
    buttons = randomized_choice_options(num_choices=2)
    prompt = 'You will play a games with 84 rounds.\nIn each round, you will be presented with 32 face-down cards.\n'\
        'Every card is either a gain card or a loss card.\nIf you turn over a gain card, the gain amount of that card (between 10 and 600 points) will be added to your current game score.\n'\
        'If you turn over a loss card, the loss amount of that card (between 25 and 750 points) will be subtracted from your game score.\n'\
        'In different rounds, between 1 and 28 cards are loss cards.\n'\
        'Loss and gain amounts also differ between rounds.\n'\
        'You may keep turning over cards as long as you keep encountering gain cards.\nYou may also stop the round at any point and claim your current payout. \n'\
        'If you encounter a loss card, the round ends immediately.\n'\
        'Your gains and losses will be summed up to give you your final score for each round.\n'

    prompt += f"Press {buttons[0]} to turn a card over, or {buttons[1]} to stop the round and claim your current payout.\n\n"

    for block in range(1, df_participant.block.max() + 1):
        df_block = df_participant[(df_participant['block'] == block)]

        for trial in df_block['trial']:
            df_trial = df_block[(df_block['trial'] == trial)]
            num_cards = df_trial.choice.item()
            n_loss_cards = df_trial.n_loss_cards.item()
            loss = df_trial.loss_value.item()
            gain = df_trial.win_value.item()
            prompt += f'Round {trial}:\n'
            prompt += f'You will be awarded {gain} points for turning over a gain card.\n'
            prompt += f'You will lose {loss} points for turning over a loss card.\nThere are {n_loss_cards} loss cards in this round.\n'

            if num_cards == 0:
                prompt += f'You press {{{buttons[1]}}} and claim your payout.\n'
            else:
                score = 0
                for i in range(num_cards - 1):
                    score += gain
                    prompt += f'You press {{{buttons[0]}}} and turn over a gain card. Your current score is {score}.\n'
                if df_trial.loss_card_encountered.item():
                    score -= loss
                    prompt += f'You press {{{buttons[0]}}} and turn over a loss card. Your current score is {score}. The round has now ended because you encountered a loss card.\n'
                else:
                    score += gain
                    prompt += f'You press {{{buttons[0]}}} and turn over a gain card. Your current score is {score}.\n'
                    prompt += f'You press {{{buttons[1]}}} and claim your payout.\n'
            prompt += f'Your final score for this round is {df_trial.reward.item()}.\n\n'
        prompt += '\n'

    prompt = prompt[:-1]
    # Final payout is a sum of final scores from three rounds selected randomly
    # prompt += f"You have completed all 84 rounds. Your final payout is {df_participant[df_participant['selected_for_payout'] == 1]['reward'].sum()}."
    print(prompt)
    all_prompts.append({'text': prompt, 'experiment': 'frey2017cct/exp1.csv', 'participant': str(participant)})

assert len(all_prompts) == df.participant.nunique(), f'The original dataset contains {df.participant.nunique()} experiments, but {len(all_prompts)} prompts have been generated.'

with jsonlines.open('prompts.jsonl', 'w') as writer:
    writer.write_all(all_prompts, ensure_ascii=False)
