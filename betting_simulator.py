
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from football_models import FootballSystem

# Configuration
INITIAL_BANKROLL = 1000.0
STAKE_SIZE = 10.0 # Fixed stake
VALUE_THRESHOLD = 0.05 # Only bet if model_prob > implied_prob + 0.05

def run_simulation(start_date='2010-08-01'):
    """
    Runs a betting simulation using the FootballSystem models.
    """
    print(f"Starting Simulation from {start_date}...")
    
    # Initialize System with default (calibrated) parameters
    fs = FootballSystem()
    
    # Load Data directly to filter for odds availability
    df = fs.load_and_prep_data()
    
    # Filter for matches with Odds
    # We rely on B365H, B365D, B365A
    mask_odds = df['B365H'].notna() & df['B365D'].notna() & df['B365A'].notna()
    
    # Filter for date (don't bet on early training data, let models warm up)
    mask_date = df['Date'] >= pd.to_datetime(start_date)
    
    # We iterate through ALL data to update ratings, but only BET on valid rows
    print(f"Total matches in dataset: {len(df)}")
    
    # Stats tracking
    bankroll = {'elo': INITIAL_BANKROLL, 'glicko': INITIAL_BANKROLL, 'trueskill': INITIAL_BANKROLL, 'poisson': INITIAL_BANKROLL}
    bets_made = {'elo': 0, 'glicko': 0, 'trueskill': 0, 'poisson': 0}
    bets_won = {'elo': 0, 'glicko': 0, 'trueskill': 0, 'poisson': 0}
    history = []
    
    for idx, row in df.iterrows():
        match_date = row['Date']
        home = row['HomeTeam']
        away = row['AwayTeam']
        result = row['FTR']
        
        # Odds
        odds_h = row['B365H']
        odds_d = row['B365D']
        odds_a = row['B365A']
        
        # Features for Updates
        # (Assuming FS handles date tracking internally if we were using its run_training, 
        # but here we must manually handle time-dependency if we want it perfect.
        # However, for simplicity let's rely on standard updates or minimal time-passing if accessible.
        # Making the manual loop robust for time-features:
        
        last_date_h = fs.teams_last_date.get(home, match_date)
        last_date_a = fs.teams_last_date.get(away, match_date)
        delta_days_h = max(0, (match_date - last_date_h).days)
        delta_days_a = max(0, (match_date - last_date_a).days)
        
        fthg = row['FTHG']
        ftag = row['FTAG']
        
        # 1. PREDICT
        # Get Current Ratings
        e_h, e_a = fs._get_team(home, 'elo'), fs._get_team(away, 'elo')
        g_h, g_a = fs._get_team(home, 'glicko'), fs._get_team(away, 'glicko')
        t_h, t_a = fs._get_team(home, 'ts'), fs._get_team(away, 'ts')
        p_h, p_a = fs._get_team(home, 'poisson'), fs._get_team(away, 'poisson')
        
        # Calculate Probabilities (Home Win)
        # Note: Models predict Home Win vs (Draw+Away). 
        # Bet365 Implied Probability for Home Win = 1 / OddsH
        
        prob_e = fs.elo_model.predict_win_prob(e_h, e_a)
        prob_g = fs.glicko_model.predict_win_prob(g_h, g_a)
        prob_t = fs.ts_model.predict_win_prob(t_h, t_a)
        prob_p = fs.poisson_model.predict_win_prob(p_h, p_a)
        
        # 2. BET
        if mask_odds[idx] and mask_date[idx]:
            implied_h = 1.0 / odds_h
            
            # Simple Strategy: Value Bet on Home Win
            # If Model Prob > Implied + Threshold
            
            # Elo
            if prob_e > implied_h + VALUE_THRESHOLD:
                bets_made['elo'] += 1
                if result == 'H':
                    profit = STAKE_SIZE * (odds_h - 1)
                    bankroll['elo'] += profit
                    bets_won['elo'] += 1
                else:
                    bankroll['elo'] -= STAKE_SIZE
            
            # Glicko
            if prob_g > implied_h + VALUE_THRESHOLD:
                bets_made['glicko'] += 1
                if result == 'H':
                    profit = STAKE_SIZE * (odds_h - 1)
                    bankroll['glicko'] += profit
                    bets_won['glicko'] += 1
                else:
                    bankroll['glicko'] -= STAKE_SIZE

            # TrueSkill
            if prob_t > implied_h + VALUE_THRESHOLD:
                bets_made['trueskill'] += 1
                if result == 'H':
                    profit = STAKE_SIZE * (odds_h - 1)
                    bankroll['trueskill'] += profit
                    bets_won['trueskill'] += 1
                else:
                    bankroll['trueskill'] -= STAKE_SIZE

            # Poisson
            if prob_p > implied_h + VALUE_THRESHOLD:
                bets_made['poisson'] += 1
                if result == 'H':
                    profit = STAKE_SIZE * (odds_h - 1)
                    bankroll['poisson'] += profit
                    bets_won['poisson'] += 1
                else:
                    bankroll['poisson'] -= STAKE_SIZE
                    
            if idx % 1000 == 0:
                history.append({
                    'date': match_date,
                    'elo': bankroll['elo'],
                    'glicko': bankroll['glicko'],
                    'ts': bankroll['trueskill'],
                    'poisson': bankroll['poisson']
                })

        # 3. UPDATE MODELS
        if result == 'H': score = 1.0
        else: score = 0.0
        
        new_e_h, new_e_a = fs.elo_model.update(e_h, e_a, score, delta_days_home=delta_days_h, delta_days_away=delta_days_a)
        new_g_h, new_g_a = fs.glicko_model.update(g_h, g_a, score, delta_days_home=delta_days_h, delta_days_away=delta_days_a)
        new_t_h, new_t_a = fs.ts_model.update(t_h, t_a, score, delta_days_home=delta_days_h, delta_days_away=delta_days_a)
        new_p_h, new_p_a = fs.poisson_model.update(p_h, p_a, score, home_goals=fthg, away_goals=ftag)
        
        fs._set_team(home, 'elo', new_e_h)
        fs._set_team(away, 'elo', new_e_a)
        fs._set_team(home, 'glicko', new_g_h)
        fs._set_team(away, 'glicko', new_g_a)
        fs._set_team(home, 'ts', new_t_h)
        fs._set_team(away, 'ts', new_t_a)
        fs._set_team(home, 'poisson', new_p_h)
        fs._set_team(away, 'poisson', new_p_a)
        
        fs.teams_last_date[home] = match_date
        fs.teams_last_date[away] = match_date
        
    print("\n--- Simulation Results ---")
    for model in ['elo', 'glicko', 'trueskill', 'poisson']:
        roi = ((bankroll[model] - INITIAL_BANKROLL) / (bets_made[model] * STAKE_SIZE)) * 100 if bets_made[model] > 0 else 0
        win_rate = (bets_won[model] / bets_made[model]) * 100 if bets_made[model] > 0 else 0
        print(f"{model.upper()}: Bets: {bets_made[model]}, Bankroll: {bankroll[model]:.2f}, ROI: {roi:.2f}%, Win Rate: {win_rate:.2f}%")

if __name__ == "__main__":
    run_simulation()
