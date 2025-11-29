import os
import sys

def enforce_sector_integrity():
    """
    IRON DOME PROTOCOL
    Ensures NFL code only runs in NFL directories.
    """
    current_path = os.getcwd()
    REQUIRED_SIGNATURE = "NFL_Sniper"
    
    if REQUIRED_SIGNATURE.lower() not in current_path.lower():
        print("\n" + "="*60)
        print("🚨 SECURITY BREACH DETECTED 🚨")
        print("="*60)
        print(f"❌ CRITICAL: NFL Code attempting to run in unauthorized sector.")
        print(f"📂 Current Sector: {current_path}")
        print("SYSTEM TERMINATED TO PREVENT DATA CORRUPTION.")
        print("="*60 + "\n")
        sys.exit(1)

# Auto-run on import
enforce_sector_integrity()