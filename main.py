import pandas as pd
import numpy as np

def variance_decomposition(player_df):

    var_obs = player_df['actual_win_pct'].var(ddof=0)
    var_rand = np.mean(player_df['sim_win_pct_stdev']**2)
    var_true = var_obs - var_rand

    return pd.Series({'var_obs': var_obs, 'var_rand': var_rand, 'var_true': var_true})

def simulate_random_fow_percent(fow_stats, n_simulations=1, random_seed=0):

    np.random.seed(random_seed)
    results = []
    for _, row in fow_stats.iterrows():
        season = row['season']
        player = row['player_id']
        n_faceoff = row['faceoffTaken']


        sim_wins = np.random.binomial(n=n_faceoff, p=0.5, size=n_simulations)
        sim_pcts = sim_wins / n_faceoff if n_faceoff > 0 else np.nan

        results.append({
            'season': season,
            'player_id': int(player),
            'faceoffTaken': n_faceoff,
            'sim_win_pct_mean': np.mean(sim_pcts),
            'sim_win_pct_std': np.std(sim_pcts),
            'sim_win_pct_all': sim_pcts if n_simulations > 1 else sim_pcts[0]
        })
    return pd.DataFrame(results)






df_faceoff = pd.read_csv('faceoffs_by_season.csv',index_col=0)
df_win = pd.read_csv('faceoff_wins_by_season.csv',index_col=0)

df = df_faceoff.merge(df_win, on=['player_id','season'], how='left')
df = df[df['season'] > 20042005]
df = df[df['faceoffTaken'] > 0]

df['actual_win_pct'] = df['faceOffWins'] / df['faceoffTaken']
sim_results = simulate_random_fow_percent(df, n_simulations=1000) #merge these back together
player_sim = (
    sim_results.groupby('player_id')
    .apply(lambda x: np.sum(x['sim_win_pct_mean'] * x['faceoffTaken']) / np.sum(x['faceoffTaken']))
    .reset_index(name='sim_win_pct_mean_weighted')
)
sim_results = player_sim[['player_id','sim_win_pct_mean_weighted']]
df = df.merge(sim_results, on='player_id', how='left')
print('boink')



# Group by player and apply the decomposition
df['sim_win_pct_stdev'] = np.sqrt(df['sim_win_pct_mean_weighted'] * (1 - df['sim_win_pct_mean_weighted']) / df['sim_win_pct_mean_weighted'])
result = df.groupby('player_id').apply(variance_decomposition).reset_index()

#print(result.head())
#print(df.head())
meaningful_count = (result['var_true'] > 0).sum()
print(f"{meaningful_count} players have a meaningful (positive) var_true.")