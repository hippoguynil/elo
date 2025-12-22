import pandas as pd
import math

df = pd.read_csv('/home/thehippo/elo/elo/data/all_data.csv')
home_wins = len(df[df['FTR'] == 'H'])
total = len(df)
prob_home = home_wins / total
prob_not_home = 1.0 - prob_home

print(f"Total Matches: {total}")
print(f"Home Wins: {home_wins} ({prob_home:.4%})")

# Calculate Dummy Log Loss
# For every match, predict prob_home
loss = 0
for idx, row in df.iterrows():
    if row['FTR'] == 'H':
        loss += -math.log(prob_home)
    else:
        loss += -math.log(prob_not_home)
        
avg_loss = loss / total
print(f"Baseline Log Loss (Always predict {prob_home:.4f}): {avg_loss:.4f}")
