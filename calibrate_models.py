import pandas as pd
from football_models import FootballSystem

def main():
    # 1. Baseline (Single forward pass)
    print("=== BASELINE (NO CALIBRATION) ===")
    system_base = FootballSystem()
    loss_e, loss_g, loss_t, loss_p = system_base.run_training(silent=False)
    
    col_1993_base = system_base.teams_elo.get('Colchester') # This is actually 2025 rating!
    # To get 1993 rating after calibration, we need to inspect the state AFTER the backward pass but before final forward pass.
    
    # 2. Calibrated
    print("\n=== CALIBRATED (3 ITERATIONS) ===")
    system_cal = FootballSystem()
    
    # We'll run one pass to store "Default" 1993 ratings (which are 1500)
    # Then calibrate
    system_cal.calibrate(iterations=3, silent=False)
    
    print("\nStarting Ratings for 1993 (Implied by final Calibration state):")
    # Actually, to truly show 1993 ratings, I need to look at the state after a BACKWARD pass.
    # The 'calibrate' method ends with a Forward pass, so the ratings in system_cal are 2025 ratings.
    
    # Let's do a manual step to see 1993 ratings
    system_show = FootballSystem()
    system_show.run_training(silent=True) # Forward
    system_show.run_training(silent=True, direction='backward') # Backward - now seeds reflect 1993
    
    teams = ['Man United', 'Liverpool', 'Arsenal', 'Colchester', 'Southend', 'Yeovil']
    print(f"{'Team':<20} | {'1993 Start (Elo)':<15}")
    print("-" * 40)
    for team in teams:
        r = system_show.teams_elo.get(team)
        val = r.rating if r else "N/A"
        print(f"{team:<20} | {val if isinstance(val, str) else f'{val:.1f}'}")

if __name__ == "__main__":
    main()
