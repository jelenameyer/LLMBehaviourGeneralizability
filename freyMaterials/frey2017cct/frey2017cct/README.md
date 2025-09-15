exp1.csv from:

Frey, R., Pedroni, A., Mata, R., Rieskamp, J., & Hertwig, R. (2017). Risk preference shares the psychometric structure of major psychological traits. Science Advances, 3, e1701381. https://doi.org/10.1126/sciadv.1701381

## Data quality checklist

| Paper | Task |
|:----------|:-------------:|
| README.md with list of original data sources in APA style format | ✔️ |
| Example prompt in README.md |  ✔️ |
| main folder name correct syntax | ✔️ |
| preprocess_data.py exists | ✔️ |
| generate_prompts.py exists | ✔️ |
| prompts.jsonl exists | ✔️ |
| expN.csv exist | ✔️ |
| original_data/ exists | ✔️ |
| expN.csv follows the template | ✔️ |
| verify that instructions match the actual experiment | ✔️ |
| Discrete choice options are randomized for each participant | ✔️ |

## Example prompt

```
You have completed all 84 rounds. Your final payout is 550.
This game will have 84 rounds. On each round, you will be presented with 32 face-down cards.
Every card is either a gain card or a loss card. If you turn over a gain card, the gain amount of that card (between 10 and 600 points) will be added to your current game score.
If you turn over a loss card, the loss amount of that card (between 25 and 750 points) will be subtracted from your game score.
On different trials, between 1 and 28 cards are loss cards.
Gain and loss amounts may differ between rounds.
On each trial, you may keep turning over cards as long as you keep encountering gain cards. You may also stop the trial at any point and claim your current payout. 
If you encounter a loss card, the trial ends.
Your gains and losses will be summed up to give you your final score for each round.
At the end of the game, three game rounds will be chosen randomly, and their scores summed up to give you your total payout.
Press '9' to turn a card over, or 'B' to stop the trial and claim your current payout.

Round 1:
In this round, you will be awarded 150 points for turning over each gain card.
You will lose 75 points for turning over a loss card.
There are 20 loss cards in this round.
- You pressed {9} and turned over a loss card over. Your current score is -75. The trial has now ended because you encountered a loss card.
Your final score for this round is -75.

Round 2:
In this round, you will be awarded 50 points for turning over each gain card.
You will lose 100 points for turning over a loss card.
There are 1 loss cards in this round.
- You pressed {9} and turned over a gain card. Your current score is 50.
- You pressed {9} and turned over a gain card. Your current score is 100.
- You pressed {9} and turned over a gain card. Your current score is 150.
- You pressed {9} and turned over a gain card. Your current score is 200.
- You pressed {9} and turned over a gain card. Your current score is 250.
- You pressed {9} and turned over a gain card. Your current score is 300.
- You pressed {9} and turned over a gain card. Your current score is 350.
- You pressed {9} and turned over a gain card. Your current score is 400.
- You pressed {9} and turned over a gain card. Your current score is 450.
- You pressed {9} and turned over a gain card. Your current score is 500.
- You pressed {9} and turned over a gain card. Your current score is 550.
- You pressed {9} and turned over a gain card. Your current score is 600.
- You pressed {9} and turned over a gain card. Your current score is 650.
- You pressed {9} and turned over a gain card. Your current score is 700.
- You pressed {9} and turned over a gain card. Your current score is 750.
- You pressed {9} and turned over a gain card. Your current score is 800.
- You pressed {9} and turned over a gain card. Your current score is 850.
- You pressed {9} and turned over a gain card. Your current score is 900.
- You pressed {9} and turned over a gain card. Your current score is 950.
- You pressed {9} and turned over a gain card. Your current score is 1000.
- You pressed {9} and turned over a gain card. Your current score is 1050.
- You pressed {9} and turned over a gain card. Your current score is 1100.
- You pressed {9} and turned over a gain card. Your current score is 1150.
- You pressed {9} and turned over a gain card. Your current score is 1200.
- You pressed {9} and turned over a gain card over. Your current score is 1250.
- You pressed {B} and claimed your payout.
Your final score for this round is 1250.

Round 3:
In this round, you will be awarded 200 points for turning over each gain card.
You will lose 100 points for turning over a loss card.
There are 10 loss cards in this round.
- You pressed {9} and turned over a gain card. Your current score is 200.
- You pressed {9} and turned over a gain card. Your current score is 400.
- You pressed {9} and turned over a gain card over. Your current score is 600.
- You pressed {B} and claimed your payout.
Your final score for this round is 600.
```
