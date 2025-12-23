import pandas as pd
import numpy as np
import math
from football_models import FootballSystem

def simulate_betting_by_div(df, model_type='glicko', min_edge=0.03):
    system = FootballSystem()
    system.reset_state(keep_ratings=False)
    
    # Trackers
    stats = {} # div -> {staked, returned, bets}
    
    print(f"Propagating and betting match-by-match (Edge > {min_edge:.0%})...")
    for idx, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        result = row['FTR']
        match_date = row['Date']
        div = row['Div']
        
        if div not in stats: stats[div] = {'staked': 0.0, 'returned': 0.0, 'bets': 0}
        
        # 1. Place Bet
        if match_date >= pd.to_datetime('2016-01-01'):
            has_odds = not (pd.isna(row['AvgH']) or pd.isna(row['AvgD']) or pd.isna(row['AvgA']))
            if has_odds:
                odd_h, odd_d, odd_a = row['AvgH'], row['AvgD'], row['AvgA']
                total_p = (1/odd_h + 1/odd_d + 1/odd_a)
                implied_h = (1/odd_h) / total_p
                
                preds = system.predict_match(home, away)
                model_p = preds[model_type]['prob_home_win']
                edge = model_p - implied_h
                
                if edge > min_edge:
                    stats[div]['staked'] += 1.0
                    stats[div]['bets'] += 1
                    if result == 'H':
                        stats[div]['returned'] += odd_h

        # 2. Update System
        if result == 'H': score = 1.0
        elif result == 'A': score = 0.0
        else: score = 0.5
        
        last_date_h = system.teams_last_date.get(home, match_date)
        last_date_a = system.teams_last_date.get(away, match_date)
        delta_h = max(0, (match_date - last_date_h).days) if last_date_h else 0
        delta_a = max(0, (match_date - last_date_a).days) if last_date_a else 0
        
        e_h, e_a = system._get_team(home, 'elo'), system._get_team(away, 'elo')
        g_h, g_a = system._get_team(home, 'glicko'), system._get_team(away, 'glicko')
        t_h, t_a = system._get_team(home, 'ts'), system._get_team(away, 'ts')
        p_h, p_a = system._get_team(home, 'poisson'), system._get_team(away, 'poisson')

        new_e_h, new_e_a = system.elo_model.update(e_h, e_a, score, delta_days_home=delta_h, delta_days_away=delta_a)
        new_g_h, new_g_a = system.glicko_model.update(g_h, g_a, score, delta_days_home=delta_h, delta_days_away=delta_a)
        new_t_h, new_t_a = system.ts_model.update(t_h, t_a, score, delta_days_home=delta_h, delta_days_away=delta_a)
        new_p_h, new_p_a = system.poisson_model.update(p_h, p_a, score, home_goals=row['FTHG'], away_goals=row['FTAG'])
        
        system._set_team(home, 'elo', new_e_h)
        system._set_team(away, 'elo', new_e_a)
        system._set_team(home, 'glicko', new_g_h)
        system._set_team(away, 'glicko', new_g_a)
        system._set_team(home, 'ts', new_t_h)
        system._set_team(away, 'ts', new_t_a)
        system._set_team(home, 'poisson', new_p_h)
        system._set_team(away, 'poisson', new_p_a)
        system.teams_last_date[home] = match_date
        system.teams_last_date[away] = match_date

    print("\n--- RESULTS BY DIVISION (2016-2025) ---")
    for div, s in stats.items():
        if s['staked'] > 0:
            roi = (s['returned'] - s['staked']) / s['staked']
            print(f"Division: {div:<3} | Bets: {s['bets']:<5} | ROI: {roi:>+7.2%}")

if __name__ == "__main__":
    df = pd.read_csv("data/all_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    simulate_betting_by_div(df, min_edge=0.05)
