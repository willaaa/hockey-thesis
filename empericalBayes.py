import pandas as pd
import numpy as np
df = pd.read_csv('faceoffs_by_season.csv',index_col=0)
df_wins = pd.read_csv('faceoff_wins_by_season.csv',index_col=0)
df = df.merge(df_wins, on=['player_id','season'])
df = df[df['season'] > 20042005]
df = df[df['faceoffTaken'] > 0]#change to groupby player?

df['p_hat'] = df['faceOffWins'] / df['faceoffTaken']

# Step 2: Compute prior
mean_p = df['p_hat'].mean()
var_p = df['p_hat'].var()

M = mean_p * (1 - mean_p) / var_p - 1
alpha = mean_p * M
beta = (1 - mean_p) * M

# Step 3: Posterior estimate for each player
df['posterior_mean'] = (df['faceOffWins'] + alpha) / (df['faceoffTaken'] + alpha + beta) #a player will win 54% of their games because of skill?
print(df.head())