import safety_protocol # Iron Dome
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime

# Silence Warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="NFL Sniper Pro", layout="wide", page_icon="🏈")
HISTORY_FILE = "nfl_betting_history.csv"

# ==========================================
# 2. INTELLIGENCE ENGINES (Backend)
# ==========================================

def log_prediction(week, home, away, winner, confidence, rating, edge, note):
    """Logs telemetry to CSV."""
    try:
        new_rec = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Week": f"Week {week}",
            "Matchup": f"{away} @ {home}",
            "Pick": winner,
            "Confidence": f"{confidence:.1f}%",
            "Edge": f"{edge:+.1f}%",
            "Rating": rating,
            "Note": note
        }
        if os.path.exists(HISTORY_FILE): df = pd.read_csv(HISTORY_FILE)
        else: df = pd.DataFrame(columns=new_rec.keys())
        df = pd.concat([df, pd.DataFrame([new_rec])], ignore_index=True)
        df.to_csv(HISTORY_FILE, index=False)
    except: pass

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
        
        # STACKING: Create Team Logs for Rolling Stats
        h_games = df[['game_id', 'season', 'week', 'home_team', 'home_score', 'home_margin']].rename(
            columns={'home_team': 'team', 'home_score': 'score', 'home_margin': 'margin'})
        a_games = df[['game_id', 'season', 'week', 'away_team', 'away_score', 'away_margin']].rename(
            columns={'away_team': 'team', 'away_score': 'score', 'away_margin': 'margin'})
        
        logs = pd.concat([h_games, a_games]).sort_values(['team', 'season', 'week'])
        
        # Rolling Calc (L5 Games)
        logs['roll_margin'] = logs.groupby('team')['margin'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        logs['roll_score'] = logs.groupby('team')['score'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        
        # Shift (Pre-Game Stats)
        logs['pre_margin'] = logs.groupby('team')['roll_margin'].shift(1).fillna(0)
        logs['pre_score'] = logs.groupby('team')['roll_score'].shift(1).fillna(0)
        
        # Merge Back
        h_stats = logs[['game_id', 'team', 'pre_margin', 'pre_score']].rename(
            columns={'team': 'home_team', 'pre_margin': 'h_mom', 'pre_score': 'h_off'})
        a_stats = logs[['game_id', 'team', 'pre_margin', 'pre_score']].rename(
            columns={'team': 'away_team', 'pre_margin': 'a_mom', 'pre_score': 'a_off'})
            
        df_final = df.merge(h_stats, on=['game_id', 'home_team'])
        df_final = df_final.merge(a_stats, on=['game_id', 'away_team'])
        
        df_final.to_csv("nfl_games_processed.csv", index=False)
        return df_final
    except Exception as e:
        st.error(f"Data Failure: {e}")
        return None

def heal_brain_engine():
    """Retrains Model on Momentum Features"""
    print("🧠 RETRAINING MOMENTUM BRAIN...")
    if not os.path.exists("nfl_games_processed.csv"): heal_data_engine()
        
    try:
        df = pd.read_csv("nfl_games_processed.csv")
        
        # Train on MOMENTUM, not Names
        features = ['h_mom', 'h_off', 'a_mom', 'a_off']
        X = df[features]
        y = df['home_win']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RidgeClassifier(alpha=1.0)
        model.fit(X_scaled, y)
        
        pkg = {"model": model, "scaler": scaler, "predictors": features}
        with open("nfl_model_v1.pkl", "wb") as f: pickle.dump(pkg, f)
        return pkg
    except Exception as e:
        st.error(f"Brain Failure: {e}")
        return None

@st.cache_resource
def load_system():
    try:
        # 1. Check Data
        if not os.path.exists("nfl_games_processed.csv"):
            with st.spinner("Downloading NFL Data..."):
                heal_data_engine()
        
        # 2. Check Brain
        if not os.path.exists("nfl_model_v1.pkl"):
            with st.spinner("Training Neural Network..."):
                heal_brain_engine()
        
        # Load Everything
        df = pd.read_csv("nfl_games_processed.csv")
        teams = sorted(df['home_team'].unique())
        
        with open("nfl_model_v1.pkl", "rb") as f:
            pkg = pickle.load(f)
            
        return df, teams, pkg
    except:
        # Force Rebuild
        heal_data_engine()
        pkg = heal_brain_engine()
        df = pd.read_csv("nfl_games_processed.csv")
        teams = sorted(df['home_team'].unique())
        return df, teams, pkg

df, teams, pkg = load_system()
model = pkg["model"]
scaler = pkg["scaler"]

# --- HELPER: GET MOMENTUM ---
def get_momentum(team_name):
    last_home = df[df['home_team'] == team_name].tail(1)
    last_away = df[df['away_team'] == team_name].tail(1)
    if not last_home.empty and (last_away.empty or last_home.index[-1] > last_away.index[-1]):
        return last_home.iloc[0]['h_mom'], last_home.iloc[0]['h_off']
    elif not last_away.empty:
        return last_away.iloc[0]['a_mom'], last_away.iloc[0]['a_off']
    return 0, 0

# ==========================================
# 3. DASHBOARD
# ==========================================
st.title("🏈 NFL SNIPER: MOMENTUM JUGGERNAUT")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📍 MATCHUP CONFIG")
    try: h_idx, a_idx = teams.index("KC"), teams.index("SF")
    except: h_idx, a_idx = 0, 1
    home_team = st.selectbox("Home Team", teams, index=h_idx)
    away_team = st.selectbox("Away Team", teams, index=a_idx)
    week = st.slider("Season Week", 1, 18, 1)
    
    st.divider()
    st.markdown("### 🎲 VEGAS CHECK")
    vegas_line = st.number_input("Enter Vegas Line (e.g. -110)", value=-110, step=10)
    
    if vegas_line < 0: imp_prob = abs(vegas_line) / (abs(vegas_line) + 100) * 100
    else: imp_prob = 100 / (vegas_line + 100) * 100
    
    st.caption(f"Implied Win Probability: {imp_prob:.1f}%")
    
    st.divider()
    st.markdown("### 📜 Betting History")
    if st.checkbox("Show History"):
        if os.path.exists(HISTORY_FILE): st.dataframe(pd.read_csv(HISTORY_FILE).tail(5), hide_index=True)
            
    with st.expander("🛠️ SYSTEM MAINTENANCE"):
        if st.button("🔄 FORCE SYSTEM UPDATE"):
            st.cache_resource.clear()
            heal_data_engine()
            heal_brain_engine()
            st.rerun()

# --- MAIN ---
c1, c2, c3 = st.columns([1, 0.5, 1])
with c1: st.markdown(f"<h1 style='text-align:center'>{away_team}</h1>", unsafe_allow_html=True)
with c2: st.markdown("<h1 style='text-align:center'>VS</h1>", unsafe_allow_html=True)
with c3: st.markdown(f"<h1 style='text-align:center'>{home_team}</h1>", unsafe_allow_html=True)

st.divider()

if st.button("🚀 RUN MOMENTUM SIMULATION", type="primary", use_container_width=True):
    h_mom, h_off = get_momentum(home_team)
    a_mom, a_off = get_momentum(away_team)
    
    in_data = pd.DataFrame([[h_mom, h_off, a_mom, a_off]], columns=['h_mom', 'h_off', 'a_mom', 'a_off'])
    sc_data = scaler.transform(in_data)
    raw = model.decision_function(sc_data)[0]
    pred = model.predict(sc_data)[0]
    
    winner = home_team if pred == 1 else away_team
    confidence = 50 + (abs(raw) * 50)
    confidence = min(99.9, confidence)
    
    edge = confidence - imp_prob
    
    rating = "PASS"
    if edge >= 10: rating = "DIAMOND"
    elif edge >= 5: rating = "GOLD"
    elif confidence > 60: rating = "SILVER"
    
    mom_note = f"Diff: {h_mom:.1f} vs {a_mom:.1f}"
    log_prediction(week, home_team, away_team, winner, confidence, rating, edge, mom_note)
    
    color = "#66fcf1" if confidence > 60 else "#c5c6c7"
    edge_color = "#00ff00" if edge > 0 else "#ff4444"
    
    c_res1, c_res2 = st.columns(2)
    with c_res1:
        st.markdown(f"""<div class="metric-card" style="border-color: {color};">
            <h3 style="color:#aaa">PROJECTED WINNER</h3>
            <h1 style="font-size:3em; margin:0; color:{color}">{winner}</h1>
            <h2 style="color:white">{confidence:.1f}%</h2>
        </div>""", unsafe_allow_html=True)
        
    with c_res2:
        st.markdown(f"""<div class="metric-card" style="border-color: {edge_color};">
            <h3 style="color:#aaa">EDGE vs VEGAS</h3>
            <h1 style="font-size:3em; margin:0; color:{edge_color}">{edge:+.1f}%</h1>
            <h2 style="color:white">{rating}</h2>
        </div>""", unsafe_allow_html=True)
        
    st.markdown("### 📊 Momentum Analysis (L5 Games)")
    st.progress(int(confidence) if winner == home_team else 100 - int(confidence))
    st.caption(f"Momentum Balance: {away_team} ⟵ ⟶ {home_team}")