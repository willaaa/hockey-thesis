# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.regression.mixed_linear_model import MixedLM
#
# # Read the data
# df = pd.read_csv('team_faceoff_overview.csv')
# #df = pd.read_csv('faceoffsByGame_filtered.csv')
#
# # Fit the mixed linear model
# model = MixedLM(endog=df['seasonFOpercent'],
#                 exog=sm.add_constant(df[['season']]),
#                 groups=df['team_id'],
#                 exog_re=df[['season']])
# result = model.fit()
#
# print("\nMixed Linear Model Results:")# should I be regressing on faceoff win percent instead?
# print(result.summary())

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# Read the data
df = pd.read_csv("team_faceoff_overview.csv")

df["season_numeric"] = pd.factorize(df["season"])[0]

# Center the season variable to improve numerical stability
df["season_centered"] = df["season_numeric"] - df["season_numeric"].mean()

# Mixed model: fixed effect of season, random intercept per team, random slope of season
model = MixedLM(
    endog=df["seasonFOpercent"],
    exog=sm.add_constant(df[["season_centered"]]),  # fixed effects
    groups=df["team_id"],                          # random intercept
    exog_re=df[["season_centered"]]                # random slope
)

# Fit the model
result = model.fit()

print("\nMixed Linear Model Results:")
print(result.summary())

import numpy as np

# Predicted values = skill
df["predicted"] = result.predict()

# Residuals = luck
df["residual"] = df["seasonFOpercent"] - df["predicted"]

# Compute variance share for each team
team_breakdown = []
for team, group in df.groupby("team_id"):
    skill_var = np.var(group["predicted"], ddof=1)
    luck_var = np.var(group["residual"], ddof=1)
    total_var = skill_var + luck_var

    skill_share = skill_var / total_var
    luck_share = luck_var / total_var

    team_breakdown.append((team, skill_share, luck_share))

# Assemble into DataFrame
team_skill_luck = pd.DataFrame(team_breakdown, columns=["team_id", "skill_share", "luck_share"])
print(team_skill_luck)
