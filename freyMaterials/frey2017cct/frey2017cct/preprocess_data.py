import pandas as pd

# Load original dataset
df = pd.read_csv("original_data/cct.csv", dtype='Int32')

out = df.filter(['partid', 'r_block', 'r_trialnum', 'r_winvalue', 'r_lossvalue',
                 'r_lossnum', 'r_cardschosen', 'r_censored', 'r_score', 'r_payout'], axis=1) # discard column with time measurements

# standardised column names
out = out.rename(columns={'partid': 'participant', 'r_trialnum': 'trial',
                          'r_block': 'block', 'r_winvalue': 'win_value', 'r_lossvalue': 'loss_value',
                          'r_censored': 'loss_card_encountered', 'r_score': 'reward', 'r_cardschosen': 'choice',
                          'r_lossnum': 'n_loss_cards',
                          'r_payout': 'selected_for_payout'})
out['task'] = 0
out.dropna(inplace=True)
# save to csv
out.to_csv("exp1.csv", index=False)
