import safety_protocol
import pandas as pd
import numpy as np
import os
import requests

print("🏈 STARTING NFL MOMENTUM REFINERY...")

URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

try:
    # 1. Download
    r = requests.get(URL)
    with open("nfl_games.csv", "wb") as f: f.write(r.content)
    
    # 2. Load & Filter
    df = pd.read_csv("nfl_games.csv")
    df = df[df['season'] >= 2020].copy()
    
    # 3. Calculate Margins
    df['home_margin'] = df['home_score'] - df['away_score']
    df['away_margin'] = df['away_score'] - df['home_score']
    df['home_win'] = np.where(df['home_margin'] > 0, 1, 0)
    
    # 4. STACKING: Create a "Team Log" to calculate rolling stats
    # We need to treat Home and Away games as just "Games Played" for a team
    home_games = df[['game_id', 'season', 'week', 'home_team', 'home_score', 'home_margin']].rename(
        columns={'home_team': 'team', 'home_score': 'score', 'home_margin': 'margin'})
    away_games = df[['game_id', 'season', 'week', 'away_team', 'away_score', 'away_margin']].rename(
        columns={'away_team': 'team', 'away_score': 'score', 'away_margin': 'margin'})
    
    team_logs = pd.concat([home_games, away_games]).sort_values(['team', 'season', 'week'])
    
    # 5. MOMENTUM CALCULATION (Last 5 Games Weighted)
    # We use a simple rolling mean of the margin (Point Differential)
    print("   -> Calculating Rolling Point Differentials (L5)...")
    team_logs['roll_margin'] = team_logs.groupby('team')['margin'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    team_logs['roll_score'] = team_logs.groupby('team')['score'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    
    # 6. MERGE BACK TO MATCHUPS
    # We map the team's rolling stats *entering* the game (shift 1 to avoid cheating)
    team_logs['pre_game_margin'] = team_logs.groupby('team')['roll_margin'].shift(1).fillna(0)
    team_logs['pre_game_score'] = team_logs.groupby('team')['roll_score'].shift(1).fillna(0)
    
    # Isolate Home and Away stats to merge back
    home_stats = team_logs[['game_id', 'team', 'pre_game_margin', 'pre_game_score']].rename(
        columns={'team': 'home_team', 'pre_game_margin': 'h_mom', 'pre_game_score': 'h_off'})
    away_stats = team_logs[['game_id', 'team', 'pre_game_margin', 'pre_game_score']].rename(
        columns={'team': 'away_team', 'pre_game_margin': 'a_mom', 'pre_game_score': 'a_off'})
    
    # Final Merge
    df_final = df.merge(home_stats, on=['game_id', 'home_team'])
    df_final = df_final.merge(away_stats, on=['game_id', 'away_team'])
    
    # Clean Columns
    cols = ['season', 'week', 'home_team', 'away_team', 'h_mom', 'h_off', 'a_mom', 'a_off', 'home_win']
    df_final = df_final[cols]
    
    df_final.to_csv("nfl_games_processed.csv", index=False)
    print(f"✅ SUCCESS: Momentum Engine Installed ({len(df_final)} Games).")

except Exception as e:
    print(f"❌ FATAL ERROR: {e}")