import pandas as pd
from football_models import FootballSystem
from datetime import datetime

def extract_match_by_match(team_name, start_date_str):
    print(f"Extracting match-by-match history for {team_name} starting from {start_date_str}...")
    start_date = pd.to_datetime(start_date_str)
    
    system = FootballSystem()
    print("Calibrating models...")
    system.calibrate(iterations=2, silent=True)
    
    # We need a manual forward pass to record match-by-match
    df = system.load_and_prep_data()
    
    history = []
    
    print("Processing matches...")
    for idx, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        result = row['FTR']
        match_date = row['Date']
        
        # --- Update System Logic (replicated) ---
        if result == 'H': score = 1.0
        elif result == 'A': score = 0.0
        else: score = 0.5
        
        fthg, ftag = row['FTHG'], row['FTAG']
        
        last_date_h = system.teams_last_date.get(home, match_date)
        last_date_a = system.teams_last_date.get(away, match_date)
        delta_h = max(0, (match_date - last_date_h).days)
        delta_a = max(0, (match_date - last_date_a).days)
        
        e_h, e_a = system._get_team(home, 'elo'), system._get_team(away, 'elo')
        g_h, g_a = system._get_team(home, 'glicko'), system._get_team(away, 'glicko')
        t_h, t_a = system._get_team(home, 'ts'), system._get_team(away, 'ts')
        p_h, p_a = system._get_team(home, 'poisson'), system._get_team(away, 'poisson')

        new_e_h, new_e_a = system.elo_model.update(e_h, e_a, score, delta_days_home=delta_h, delta_days_away=delta_a)
        new_g_h, new_g_a = system.glicko_model.update(g_h, g_a, score, delta_days_home=delta_h, delta_days_away=delta_a)
        new_t_h, new_t_a = system.ts_model.update(t_h, t_a, score, delta_days_home=delta_h, delta_days_away=delta_a)
        new_p_h, new_p_a = system.poisson_model.update(p_h, p_a, score, home_goals=fthg, away_goals=ftag)
        
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
        
        # --- Recording ---
        if (home == team_name or away == team_name) and match_date >= start_date:
            r_elo = system.teams_elo.get(team_name).rating
            r_glicko = system.teams_glicko.get(team_name).rating
            r_ts = system.teams_ts.get(team_name).mu
            p_r = system.teams_poisson.get(team_name)
            
            res_val = result if home == team_name else ('H' if result == 'A' else ('A' if result == 'H' else 'D'))
            
            history.append({
                'Date': match_date,
                'Opponent': away if home == team_name else home,
                'Venue': 'H' if home == team_name else 'A',
                'Result': res_val,
                'Elo': r_elo,
                'Glicko': r_glicko,
                'TrueSkill': r_ts,
                'Poisson_Att': p_r.attack,
                'Poisson_Def': p_r.defense
            })
            
    return history

if __name__ == "__main__":
    # Danny Cowley started Jan 4, 2024. 
    # 2 seasons before would be start of 2021/22 season (Aug 2021).
    match_history = extract_match_by_match("Colchester", "2021-08-01")
    res_df = pd.DataFrame(match_history)
    res_df.to_csv("data/colchester_match_history.csv", index=False)
    print("Match history saved to data/colchester_match_history.csv")
