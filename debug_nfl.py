import pandas as pd
import safety_protocol

print("🕵️ RUNNING DIAGNOSTIC...")

try:
    df = pd.read_csv("nfl_games_processed.csv")
    print(f"✅ Data Loaded: {len(df)} games")
    
    # Check if Momentum Columns have variance
    print("\n📊 MOMENTUM CHECK:")
    print(df[['h_mom', 'a_mom']].describe())
    
    # Check specific teams
    kc = df[df['home_team'] == 'KC'].tail(1)
    chi = df[df['home_team'] == 'CHI'].tail(1)
    
    print(f"\nKC Momentum: {kc['h_mom'].values[0]}")
    print(f"CHI Momentum: {chi['h_mom'].values[0]}")
    
    if kc['h_mom'].values[0] == 0 and chi['h_mom'].values[0] == 0:
        print("❌ FAILURE: Momentum is ZERO for everyone. The Rolling Calculation failed.")
    else:
        print("✅ SUCCESS: Teams have unique momentum.")

except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}")