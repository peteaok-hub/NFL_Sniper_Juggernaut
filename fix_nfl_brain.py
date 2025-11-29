import safety_protocol
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

print("🧠 TRAINING MOMENTUM MODEL...")

if not os.path.exists("nfl_games_processed.csv"): exit()

try:
    df = pd.read_csv("nfl_games_processed.csv")
    
    # Features: Home Momentum, Home Offense, Away Momentum, Away Offense
    # We remove Team Names from training so the model learns "Stats", not "Bias"
    features = ['h_mom', 'h_off', 'a_mom', 'a_off']
    target = 'home_win'
    
    X = df[features]
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RidgeClassifier(alpha=1.0)
    model.fit(X_scaled, y)
    
    # Save Package
    pkg = {"model": model, "scaler": scaler, "predictors": features}
    with open("nfl_model_v1.pkl", "wb") as f:
        pickle.dump(pkg, f)
        
    print("✅ SUCCESS: Momentum Logic Learned.")

except Exception as e: print(f"Error: {e}")