# NBA SNIPER INTELLIGENCE ENGINE V9.1 (REST & FATIGUE MASTER)
# STATUS: MANUAL ENTRY + FATIGUE ALGORITHM
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler

# CONFIG
DATA_FILE = "nba_games_processed.csv"
MODEL_FILE = "nba_model_v1.pkl"
HISTORY_FILE = "nba_betting_ledger.csv"

# --- 1. DATA ENGINE ---
def update_nba_data():
    if not os.path.exists(DATA_FILE):
        cols = ['game_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score', 'h_mom', 'a_mom', 'h_off', 'a_off']
        df = pd.DataFrame(columns=cols)
        df.to_csv(DATA_FILE, index=False)
        return df
    return pd.read_csv(DATA_FILE)

# --- 2. BRAIN ENGINE ---
def train_nba_model():
    if not os.path.exists(DATA_FILE): update_nba_data()
    try:
        model_win = RidgeClassifier()
        model_spread = Ridge()
        model_total = Ridge()
        scaler = StandardScaler()
        
        X_mock = np.array([[0,0], [10,10]])
        y_win = np.array([0, 1])
        y_spread = np.array([-5, 5])
        y_total = np.array([200, 220])
        
        model_win.fit(X_mock, y_win)
        model_spread.fit(X_mock, y_spread)
        model_total.fit(X_mock, y_total)
        
        pkg = {"model_win": model_win, "model_spread": model_spread, "model_total": model_total, "scaler": scaler}
        with open(MODEL_FILE, "wb") as f: pickle.dump(pkg, f)
        return pkg
    except: return None

def load_brain_engine():
    if not os.path.exists(DATA_FILE): update_nba_data()
    if not os.path.exists(MODEL_FILE): train_nba_model()
    try:
        with open(MODEL_FILE, "rb") as f: pkg = pickle.load(f)
        return pd.read_csv(DATA_FILE), pkg
    except:
        pkg = train_nba_model()
        return pd.read_csv(DATA_FILE), pkg

# --- 3. TARGETING FEED (DEC 7 SLATE) ---
def get_todays_games():
    """
    MANUAL ENTRY ZONE (DEC 7).
    CRITICAL: Update 'rest' values based on yesterday's games!
    0 = Played Yesterday (B2B)
    1 = 1 Day Rest
    """
    return [
        # 12:00 PM
        {
            "home": "NYK", "away": "ORL", "time": "12:00 PM", 
            "h_rec": "15-7", "h_rest": 1, # Verify Rest
            "a_rec": "14-9", "a_rest": 1, 
            "book_spread": -4.0, "spread_odds": -110, 
            "book_total": 232.0, "total_odds": -110,
            "h_ml": -170, "a_ml": +140
        },

        # 3:30 PM
        {
            "home": "TOR", "away": "BOS", "time": "3:30 PM", 
            "h_rec": "15-9", "h_rest": 1, 
            "a_rec": "14-9", "a_rest": 1, 
            "book_spread": 2.5, "spread_odds": -105, # Celtics -2.5
            "book_total": 226.5, "total_odds": -110,
            "h_ml": +120, "a_ml": -140
        },

        # 6:00 PM
        {
            "home": "CHA", "away": "DEN", "time": "6:00 PM", 
            "h_rec": "7-16", "h_rest": 1,
            "a_rec": "16-6", "a_rest": 1, 
            "book_spread": 10.5, "spread_odds": -110, # Nuggets -10.5
            "book_total": 233.5, "total_odds": -110,
            "h_ml": +375, "a_ml": -500
        },
        {
            "home": "MEM", "away": "POR", "time": "6:00 PM", 
            "h_rec": "10-13", "h_rest": 1,
            "a_rec": "9-14", "a_rest": 0, # Blazers on B2B (Played DET yesterday)
            "book_spread": -1.0, "spread_odds": -110, 
            "book_total": 233.5, "total_odds": -110,
            "h_ml": -115, "a_ml": -105
        },

        # 7:00 PM
        {
            "home": "CHI", "away": "GSW", "time": "7:00 PM", 
            "h_rec": "9-13", "h_rest": 1,
            "a_rec": "12-12", "a_rest": 1, 
            "book_spread": 1.0, "spread_odds": -110, # Warriors -1
            "book_total": 227.5, "total_odds": -110,
            "h_ml": -105, "a_ml": -115
        },

        # 7:30 PM
        {
            "home": "PHI", "away": "LAL", "time": "7:30 PM", 
            "h_rec": "13-9", "h_rest": 1,
            "a_rec": "16-6", "a_rest": 1, 
            "book_spread": 4.0, "spread_odds": -110, # Lakers -4
            "book_total": 235.5, "total_odds": -110,
            "h_ml": +145, "a_ml": -175
        },
        
        # 8:00 PM
        {
            "home": "UTA", "away": "OKC", "time": "8:00 PM", 
            "h_rec": "8-14", "h_rest": 1,
            "a_rec": "22-1", "a_rest": 1, 
            "book_spread": 10.5, "spread_odds": -110, # Thunder -10.5
            "book_total": 239.0, "total_odds": -110,
            "h_ml": +350, "a_ml": -475
        },
    ]

# --- 4. PREDICTION LOGIC (THE CRUSHER V9.1) ---
def get_matchup_projection(game_data, away_team_unused=None):
    # HANDLING HYBRID INPUT (Works with both V6 and V9 app.py)
    if isinstance(game_data, dict):
        home = game_data['home']
        away = game_data['away']
    else:
        home = game_data
        away = away_team_unused
        game_data = {'home': home, 'away': away} # Fallback for old apps

    h_rat = 5
    a_rat = 5
    
    # Power Rankings (Tiered)
    tier_1 = ["OKC", "BOS", "DEN", "LAL", "CLE", "HOU"] 
    tier_2 = ["NYK", "ORL", "MEM", "DAL", "GSW", "TOR", "PHI"] 
    tier_3 = ["MIN", "MIA", "LAC", "PHX", "MIL", "ATL", "CHI"] 
    tier_4 = ["IND", "DET", "POR", "SAS", "CHA", "UTA", "WAS", "NOP", "SAC", "BKN"]
    
    def get_rating(team):
        if team in tier_1: return 10
        if team in tier_2: return 6
        if team in tier_3: return 2
        return -5 

    h_rat = get_rating(home) + 3 # Home Court
    a_rat = get_rating(away)
    
    # --- FATIGUE ENGINE ---
    # Penalty for 0 days rest (Back-to-Back)
    h_rest = game_data.get('h_rest', 1)
    a_rest = game_data.get('a_rest', 1)
    
    if h_rest == 0: h_rat -= 4.0 # Tired legs penalty
    if a_rest == 0: a_rat -= 5.0 # Road B2B penalty (Severe)
    
    # --- CALCULATION ---
    raw_spread = a_rat - h_rat 
    win_prob = 1 / (1 + np.exp(0.15 * raw_spread)) * 100
    
    # Visuals
    h_emoji = "🔥" if get_rating(home) >= 6 else "⚠️" if h_rest == 0 else ""
    a_emoji = "🔥" if get_rating(away) >= 6 else "⚠️" if a_rest == 0 else ""
    
    # Scoring
    base_total = 230
    if h_rest == 0 or a_rest == 0: base_total -= 6 # Tired teams shoot worse
    if home in tier_1 or away in tier_1: base_total -= 2
    
    proj_h_score = (base_total / 2) - (raw_spread / 2)
    proj_a_score = (base_total / 2) + (raw_spread / 2)
    
    return {
        "projected_spread": raw_spread,
        "projected_total": base_total,
        "win_prob": win_prob,
        "score_str": f"{int(proj_a_score)} - {int(proj_h_score)}",
        "h_emoji": h_emoji,
        "a_emoji": a_emoji
    }

def log_transaction(matchup, pick, wager, result="Pending"):
    try:
        new_rec = {"Date": datetime.now().strftime("%Y-%m-%d %H:%M"), "Matchup": matchup, "Pick": pick, "Wager": wager, "Result": result}
        if os.path.exists(HISTORY_FILE): df = pd.read_csv(HISTORY_FILE)
        else: df = pd.DataFrame(columns=new_rec.keys())
        df = pd.concat([df, pd.DataFrame([new_rec])], ignore_index=True)
        df.to_csv(HISTORY_FILE, index=False)
    except: pass