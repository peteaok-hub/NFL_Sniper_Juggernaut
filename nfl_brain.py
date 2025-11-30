import pandas as pd
import numpy as np
import requests
import os
import pickle
from datetime import datetime
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

# CONFIG
HISTORY_FILE = "nfl_betting_history.csv"
DATA_FILE = "nfl_games_processed.csv"
MODEL_FILE = "nfl_model_v1.pkl"

def heal_data_engine():
    """Downloads Data & Calculates Momentum"""
    print("📡 DOWNLOADING FRESH NFL DATA...")
    URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
    try:
        r = requests.get(URL, timeout=10)
        with open("nfl_games.csv", "wb") as f: f.write(r.content)
        
        # Load & Filter
        df = pd.read_csv("nfl_games.csv")
        df = df[df['season'] >= 2020].copy()
        df['home_margin'] = df['home_score'] - df['away_score']
        df['away_margin'] = df['away_score'] - df['home_score']
        df['home_win'] = np.where(df['home_margin'] > 0, 1, 0)
        
        # STACKING
        h_games = df[['game_id', 'season', 'week', 'home_team', 'home_score', 'home_margin', 'gameday', 'gametime']].rename(
            columns={'home_team': 'team', 'home_score': 'score', 'home_margin': 'margin'})
        a_games = df[['game_id', 'season', 'week', 'away_team', 'away_score', 'away_margin', 'gameday', 'gametime']].rename(
            columns={'away_team': 'team', 'away_score': 'score', 'away_margin': 'margin'})
        
        logs = pd.concat([h_games, a_games]).sort_values(['team', 'season', 'week'])
        
        # ROLLING MOMENTUM (L5)
        logs['roll_margin'] = logs.groupby('team')['margin'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        logs['roll_score'] = logs.groupby('team')['score'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        
        # PRE-GAME SHIFT
        logs['pre_margin'] = logs.groupby('team')['roll_margin'].shift(1).fillna(0)
        logs['pre_score'] = logs.groupby('team')['roll_score'].shift(1).fillna(0)
        
        # MERGE
        h_stats = logs[['game_id', 'team', 'pre_margin', 'pre_score']].rename(
            columns={'team': 'home_team', 'pre_margin': 'h_mom', 'pre_score': 'h_off'})
        a_stats = logs[['game_id', 'team', 'pre_margin', 'pre_score']].rename(
            columns={'team': 'away_team', 'pre_margin': 'a_mom', 'pre_score': 'a_off'})
            
        df_final = df.merge(h_stats, on=['game_id', 'home_team'])
        df_final = df_final.merge(a_stats, on=['game_id', 'away_team'])
        
        df_final.to_csv(DATA_FILE, index=False)
        return df_final
    except Exception as e:
        print(f"Data Failure: {e}")
        return None

def heal_brain_engine():
    """Retrains Model"""
    print("🧠 RETRAINING MOMENTUM BRAIN...")
    if not os.path.exists(DATA_FILE): heal_data_engine()
        
    try:
        df = pd.read_csv(DATA_FILE)
        features = ['h_mom', 'h_off', 'a_mom', 'a_off']
        X = df[features]
        y = df['home_win']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RidgeClassifier(alpha=1.0)
        model.fit(X_scaled, y)
        
        pkg = {"model": model, "scaler": scaler, "predictors": features}
        with open(MODEL_FILE, "wb") as f: pickle.dump(pkg, f)
        return pkg
    except: return None

def load_system():
    if not os.path.exists(DATA_FILE): heal_data_engine()
    if not os.path.exists(MODEL_FILE): heal_brain_engine()
    
    try:
        df = pd.read_csv(DATA_FILE)
        teams = sorted(df['home_team'].unique())
        with open(MODEL_FILE, "rb") as f: pkg = pickle.load(f)
        return df, teams, pkg
    except:
        return None, None, None

def get_momentum(df, team_name):
    last_home = df[df['home_team'] == team_name].tail(1)
    last_away = df[df['away_team'] == team_name].tail(1)
    
    if not last_home.empty and (last_away.empty or last_home.index[-1] > last_away.index[-1]):
        return last_home.iloc[0]['h_mom'], last_home.iloc[0]['h_off']
    elif not last_away.empty:
        return last_away.iloc[0]['a_mom'], last_away.iloc[0]['a_off']
    return 0, 0

def get_weekly_schedule(df, week):
    """Fetches the schedule for a specific week to build the War Grid."""
    try:
        # Find the latest season in the data
        current_season = df['season'].max()
        
        # Filter for that season and week
        schedule = df[(df['season'] == current_season) & (df['week'] == week)]
        # Drop duplicates if multiple entries exist per game
        schedule = schedule.drop_duplicates(subset=['game_id'])
        
        games = []
        for _, row in schedule.iterrows():
            games.append({
                "home": row['home_team'],
                "away": row['away_team'],
                "day": row.get('gameday', 'TBD'),
                "time": row.get('gametime', 'TBD')
            })
        return games
    except: return []

# --- FINANCIAL ENGINES ---
def calculate_kelly(bankroll, win_prob, odds_decimal, fractional=0.25):
    """Kelly Criterion for Bet Sizing."""
    if odds_decimal <= 1: return 0, 0
    b = odds_decimal - 1
    p = win_prob / 100
    q = 1 - p
    f = (b * p - q) / b
    safe_f = max(0, f) * fractional
    wager = bankroll * safe_f
    return wager, safe_f * 100

def fetch_best_odds(api_key, home, away):
    """Placeholder for Odds API - Returns structure expected by UI"""
    return [] # Returns empty list so UI doesn't crash if API fails

def log_prediction(week, home, away, winner, confidence, rating, edge, note, wager="0", book="N/A"):
    try:
        new_rec = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Week": f"Week {week}",
            "Matchup": f"{away} @ {home}",
            "Pick": winner,
            "Conf": f"{confidence:.1f}%",
            "Edge": f"{edge:+.1f}%",
            "Rating": rating,
            "Wager": wager,
            "Book": book,
            "Note": note
        }
        if os.path.exists(HISTORY_FILE): df = pd.read_csv(HISTORY_FILE)
        else: df = pd.DataFrame(columns=new_rec.keys())
        df = pd.concat([df, pd.DataFrame([new_rec])], ignore_index=True)
        df.to_csv(HISTORY_FILE, index=False)
    except: pass