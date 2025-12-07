import streamlit as st
import pandas as pd
import nba_brain as brain
import importlib

# --- BRAIN SHOCK PROTOCOL ---
importlib.reload(brain)
# ---------------------------

st.set_page_config(page_title="SNIPER V9.0", page_icon="🏀", layout="wide")

# STYLES (Hard Rock Red/Dark Theme)
st.markdown("""
<style>
    .stApp { background-color: #0e0e10; color: #e0e0e0; }
    .hr-header { 
        background: linear-gradient(90deg, #d50000, #ff1744); 
        padding: 12px; border-radius: 8px; text-align: center; 
        color: white; font-weight: 900; letter-spacing: 1px; margin-bottom: 20px;
    }
    .game-card { 
        background-color: #1c1c1e; border: 1px solid #333; padding: 15px; 
        border-radius: 12px; margin-bottom: 12px;
    }
    .edge-box { 
        background-color: #2c2c2e; border-radius: 6px; padding: 8px; text-align: center; 
        border: 1px solid #444; height: 100%;
    }
    .sniper-val { font-size: 1.2em; font-weight: bold; }
    .green-edge { color: #00e676; text-shadow: 0 0 5px rgba(0, 230, 118, 0.4); }
    .white-edge { color: #ffffff; }
    div.stButton > button { background-color: #d50000; color: white; border: none; width: 100%; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# LOAD BRAIN
try:
    df_games, pkg = brain.load_brain_engine()
except Exception as e:
    st.error(f"Brain Offline: {e}")
    st.stop()

# SIDEBAR
with st.sidebar:
    st.title("🏀 SNIPER V9.0")
    st.caption("Fatigue & Rest Protocol")
    if st.button("🔄 Refresh"): st.rerun()
    st.info("🔥 = Winning Streak | ⚠️ = Back-to-Back (Tired)")

# HEADER
st.markdown('<div class="hr-header">THE CRUSHER BOARD</div>', unsafe_allow_html=True)

# GAME GRID
games = brain.get_todays_games()

c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
c1.caption("MATCHUP & REST")
c2.caption("SPREAD EDGE")
c3.caption("TOTAL EDGE")
c4.caption("ACTION")

for g in games:
    try:
        # --- V9.0 FIX: Pass the WHOLE game object 'g' ---
        proj = brain.get_matchup_projection(g) 
    except Exception as e:
        st.error(f"Error projecting {g['home']} vs {g['away']}: {e}")
        continue
    
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        cols = st.columns([1.5, 1, 1, 1])
        
        # 1. MATCHUP
        with cols[0]:
            h_rest_txt = "B2B" if g.get('h_rest') == 0 else f"{g.get('h_rest', 1)}d Rest"
            a_rest_txt = "B2B" if g.get('a_rest') == 0 else f"{g.get('a_rest', 1)}d Rest"
            
            st.markdown(f"**{g['away']}** {proj.get('a_emoji','')} @ **{g['home']}** {proj.get('h_emoji','')}", unsafe_allow_html=True)
            st.caption(f"Rest: {a_rest_txt} vs {h_rest_txt}")
            st.caption(f"Proj Score: {proj['score_str']}")
            
        # 2. SPREAD
        with cols[1]:
            my_spread = round(proj['projected_spread'], 1)
            book_spread = g.get('book_spread', 0)
            diff = abs(my_spread - book_spread)
            color = "green-edge" if diff >= 3.0 else "white-edge"
            
            st.markdown(f"""
            <div class='edge-box'>
                <div class='sniper-val {color}'>{my_spread}</div>
                <div class='book-val'>Book: {book_spread}</div>
            </div>
            """, unsafe_allow_html=True)

        # 3. TOTAL
        with cols[2]:
            my_total = round(proj['projected_total'], 1)
            book_total = g.get('book_total', 220)
            diff_t = abs(my_total - book_total)
            color_t = "green-edge" if diff_t >= 5.0 else "white-edge"
            
            st.markdown(f"""
            <div class='edge-box'>
                <div class='sniper-val {color_t}'>{my_total}</div>
                <div class='book-val'>Book: {book_total}</div>
            </div>
            """, unsafe_allow_html=True)

        # 4. ACTION
        with cols[3]:
            winner = g['home'] if proj['win_prob'] > 50 else g['away']
            conf = int(proj['win_prob']) if winner == g['home'] else int(100 - proj['win_prob'])
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"Bet {winner}", key=f"b_{g['home']}"):
                brain.log_transaction(f"{g['away']} @ {g['home']}", winner, "1u")
                st.toast(f"✅ Logged: {winner}")
            st.caption(f"Conf: {conf}%")

        st.markdown('</div>', unsafe_allow_html=True)