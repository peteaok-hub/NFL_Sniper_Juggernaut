import streamlit as st
import pandas as pd
import nfl_brain as brain
import warnings
import os

warnings.filterwarnings("ignore")

st.set_page_config(page_title="NFL Sniper 5.0", layout="wide", page_icon="🏈")

# CONFIG
ODDS_API_KEY = "3e039d8cfd426d394b020b55bd303a07"
TEAM_MAP = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LA","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
    "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS"
}

# STYLES
st.markdown("""
<style>
    .stApp { background-color: #0e0e12; color: #ffffff; font-family: monospace; }
    .metric-card {
        background: #1f1f1f; border: 1px solid #444; border-radius: 8px; padding: 15px;
        text-align: center; margin-bottom: 10px;
    }
    .best-line { border: 2px solid #00ff00; animation: pulse 2s infinite; }
    .arb-alert { background: #ffdd00; color: black; font-weight: bold; padding: 5px; border-radius: 5px; }
    h1, h2, h3 { color: white !important; }
    .neon-green { color: #00ff00; font-weight: bold; }
    .neon-red { color: #ff3355; font-weight: bold; }
    
    div.stButton > button {
        width: 100%; background-color: #222; color: white; border: 1px solid #444; height: 80px;
    }
    div.stButton > button:hover { border-color: #00ffaa; color: #00ffaa; }
</style>
""", unsafe_allow_html=True)

# LOAD
df, teams, pkg = brain.load_system()
if not pkg: st.error("⚠️ System Offline."); st.stop()
model, scaler = pkg["model"], pkg["scaler"]

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏦 KELLY VAULT")
    
    bankroll = st.number_input("Total Bankroll ($)", value=1000.0, step=100.0)
    kelly_fraction = st.selectbox("Risk Profile", [0.25, 0.5, 1.0], format_func=lambda x: f"{x}x Kelly (Recommended: 0.25)")
    
    st.divider()
    st.markdown("### 📍 MISSION CONTROL")
    max_week = int(df['week'].max()) if 'week' in df.columns else 18
    if "week" not in st.session_state: st.session_state.week = max_week
    selected_week = st.slider("Week", 1, 18, st.session_state.week)
    
    if st.button("🔄 REFRESH MARKET"):
        st.cache_resource.clear()
        st.rerun()
        
    if st.checkbox("Show Ledger"):
        if os.path.exists(brain.HISTORY_FILE):
            st.dataframe(pd.read_csv(brain.HISTORY_FILE).tail(5), hide_index=True)

# --- WAR GRID ---
st.markdown(f"### 📅 WEEK {selected_week} SCHEDULE")
schedule = brain.get_weekly_schedule(df, selected_week)

if schedule:
    cols = st.columns(4)
    for i, game in enumerate(schedule):
        with cols[i % 4]:
            if st.button(f"{game['away']} @ {game['home']}\n{game['day']}", key=f"g_{i}"):
                st.session_state.home = game['home']
                st.session_state.away = game['away']
                st.session_state.week = selected_week
                st.rerun()

st.divider()

# --- COMMAND DECK ---
if "home" in st.session_state:
    home, away = st.session_state.home, st.session_state.away
    
    # 1. PREDICTION
    h_mom, h_off = brain.get_momentum(df, home)
    a_mom, a_off = brain.get_momentum(df, away)
    in_data = pd.DataFrame([[h_mom, h_off, a_mom, a_off]], columns=['h_mom', 'h_off', 'a_mom', 'a_off'])
    sc_data = scaler.transform(in_data)
    raw = model.decision_function(sc_data)[0]
    pred = model.predict(sc_data)[0]
    
    winner = home if pred == 1 else away
    confidence = 50 + (abs(raw) * 50)
    confidence = min(99.9, confidence)
    
    # 2. LINE SHOPPING (LIVE API)
    # Fetch odds for ALL games and find ours
    # Note: Using fetch_best_odds logic (implementation assumed in nfl_brain or inline)
    # For this snippet, we'll simulate the call or need the function in nfl_brain.py as well
    # Assuming fetch_best_odds is not in nfl_brain.py yet based on previous turn, let's put it inline here or add to brain.
    # To keep it clean, I'll add the logic here using the brain module if available, or implement a local helper.
    
    # Helper to fetch odds locally if not in brain
    def get_odds_local(h, a):
         url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={ODDS_API_KEY}&regions=us&markets=h2h&bookmakers=draftkings,fanduel,betmgm,caesars&oddsFormat=american"
         try:
             return requests.get(url).json()
         except: return []

    raw_odds = get_odds_local(home, away)
    best_line = {"book": "Unknown", "odds": -110, "dec": 1.91}
    
    # Parse API Response to find BEST odds for the PREDICTED WINNER
    for g in raw_odds:
        # Fuzzy match teams
        api_h = TEAM_MAP.get(g['home_team'], g['home_team'])
        api_a = TEAM_MAP.get(g['away_team'], g['away_team'])
        
        if api_h == home or api_a == away:
            for book in g['bookmakers']:
                for m in book['markets']:
                    if m['key'] == 'h2h':
                        for outcome in m['outcomes']:
                            # Map outcome name to code
                            o_name = TEAM_MAP.get(outcome['name'], outcome['name'][:3].upper())
                            
                            if o_name == winner:
                                # Check if this is the best price found so far
                                if outcome['price'] > best_line['dec']:
                                    best_line = {
                                        "book": book['title'],
                                        "odds": outcome['point'] if 'point' in outcome else outcome['price'], # Handle US/Dec format issues
                                        "dec": outcome['price']
                                    }
                                    # Convert decimal to American for display if needed
                                    if best_line['odds'] < 2.0 and best_line['odds'] > 1.0:
                                         # Simple Decimal to American approx for display
                                         if best_line['dec'] >= 2.0: best_line['amer'] = f"+{int((best_line['dec']-1)*100)}"
                                         else: best_line['amer'] = f"{int(-100/(best_line['dec']-1))}"
                                    else:
                                         best_line['amer'] = str(best_line['odds'])

    # 3. KELLY CALCULATION
    # Inline Kelly calc if not in brain
    def calc_kelly(bank, prob, dec_odds, frac):
        if dec_odds <= 1: return 0, 0
        b = dec_odds - 1
        p = prob / 100
        q = 1 - p
        f = (b * p - q) / b
        safe_f = max(0, f) * frac
        return bank * safe_f, safe_f * 100

    wager_amt, kelly_pct = calc_kelly(bankroll, confidence, best_line['dec'], kelly_fraction)
    
    # 4. RATING
    imp_prob = (1/best_line['dec']) * 100
    edge = confidence - imp_prob
    rating = "PASS"
    if edge > 10: rating = "💎 DIAMOND"
    elif edge > 5: rating = "🥇 GOLD"
    elif confidence > 60: rating = "🥈 SILVER"

    # DISPLAY
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card" style="border-color:#00ffaa">
            <h3>WINNER</h3><h1 style="color:#00ffaa">{winner}</h1>
        </div>""", unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""<div class="metric-card">
            <h3>BEST LINE</h3>
            <h1 style="color:white">{best_line['amer']}</h1>
            <p style="color:#aaa">@ {best_line['book']}</p>
        </div>""", unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""<div class="metric-card" style="border-color:#ffcc00">
            <h3>KELLY WAGER</h3>
            <h1 style="color:#ffcc00">${wager_amt:.2f}</h1>
            <p>{kelly_pct:.1f}% of Bank</p>
        </div>""", unsafe_allow_html=True)
        
    # LOGGING
    if st.button("💾 LOG TRANSACTION"):
        brain.log_prediction(st.session_state.week, home, away, winner, confidence, rating, edge, f"Kelly {kelly_pct:.1f}%")
        st.success("Transaction Recorded.")
    
    # MOMENTUM
    st.progress(int(confidence) if winner == home else 100 - int(confidence))
    st.caption(f"Momentum: {away} ({a_mom:.1f}) <--> {home} ({h_mom:.1f}) | Edge: {edge:.1f}%")