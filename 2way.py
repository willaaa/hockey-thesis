import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

df = pd.read_csv('this_time_for_real.csv')
#df = pd.read_csv('faceoffsByGame_filtered.csv')

df["season_numeric"] = pd.factorize(df["season"])[0]

# Center the season variable to improve numerical stability
df["season_centered"] = df["season_numeric"] - df["season_numeric"].mean()

df['winPercent'] = df['faceOffWins'] / df['faceoffTaken']
# Fit the mixed linear model
model = MixedLM(endog=df['winPercent'],
                exog=sm.add_constant(df[['season_centered']]),
                groups=df['player_id'],
                exog_re=df[['season_centered']])
result = model.fit()

# Print the summary
print("\nMixed Linear Model Results:")# should I be regressing on faceoff win percent instead?
print(result.summary())

skill = result.random_effects  # Dictionary: player_id → effect(s)
import numpy as np

df["predicted"] = result.predict()

# Residuals = luck
df["residual"] = df["winPercent"] - df["predicted"]

team_breakdown = []
for team, group in df.groupby("player_id"):
    skill_var = np.var(group["predicted"], ddof=1)
    luck_var = np.var(group["residual"], ddof=1)
    total_var = skill_var + luck_var

    skill_share = skill_var / total_var
    luck_share = luck_var / total_var

    team_breakdown.append((team, skill_share, luck_share))

# Assemble into DataFrame
team_skill_luck = pd.DataFrame(team_breakdown, columns=["player_id", "skill_share", "luck_share"])
print(team_skill_luck)


# #Luck: Calculate residuals (observation-specific random noise)
# df['predicted'] = result.predict()
# df['luck'] = df['faceOffWins'] - df['predicted']
# print(df.head())
# # Read the data
