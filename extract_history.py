import pandas as pd
import math
from football_models import FootballSystem

def calculate_ranks(points, goal_diff, goals_for, team_divs):
    """Calculates global 1-92 rank based on league finish."""
    # Organize by division
    div_teams = {}
    for team, div in team_divs.items():
        if div not in div_teams: div_teams[div] = []
        div_teams[div].append(team)
    
    # Sort order of divisions: E0, E1, E2, E3
    sorted_divs = ['E0', 'E1', 'E2', 'E3']
    global_rank = {}
    current_rank = 1
    
    for div in sorted_divs:
        if div not in div_teams: continue
        # Sort teams in this division
        teams = div_teams[div]
        teams.sort(key=lambda x: (points.get(x, 0), goal_diff.get(x, 0), goals_for.get(x, 0)), reverse=True)
        
        for team in teams:
            global_rank[team] = current_rank
            current_rank += 1
            
    return global_rank

def get_snapshot(system, target_teams, season, global_ranks):
    snapshots = []
    for team in target_teams:
        elo = system.teams_elo.get(team, system.elo_model.get_initial_rating()).rating
        glicko = system.teams_glicko.get(team, system.glicko_model.get_initial_rating()).rating
        ts = system.teams_ts.get(team, system.ts_model.get_initial_rating()).mu  # Mu is the rating
        p_r = system.teams_poisson.get(team, system.poisson_model.get_initial_rating())
        pos = global_ranks.get(team, 92) # Default to bottom if not found
        
        snapshots.append({
            'Season': season,
            'Team': team,
            'Elo': elo,
            'Glicko': glicko,
            'TrueSkill': ts,
            'Poisson_Att': p_r.attack,
            'Poisson_Def': p_r.defense,
            'Pos': pos
        })
    return snapshots

def extract_team_history(target_teams):
    print(f"Extracting history for: {target_teams}...")
    system = FootballSystem()
    
    # Use calibration to get high-fidelity starting state for 1993
    # Note: We stop before the final forward pass to extract year-by-year data ourselves
    print("Calibrating starting ratings...")
    system.calibrate(iterations=2, silent=True)
    # The calibrate method ends with a run_training(direction='forward'). 
    # To get snapshots, we need to reset to the state AFTER the backward pass but BEFORE the forward pass.
    # Actually, let's just run calibrate and then re-run one manual forward pass.
    system.run_training(silent=True, direction='forward') # Force final state
    # Wait, the above is wrong. Calibrate already does a forward pass. 
    # Let's just do: Forward -> Backward -> MANUAL Forward (with snapshots)
    system.run_training(silent=True, direction='forward')
    system.run_training(silent=True, direction='backward') 
    
    df = system.load_and_prep_data()
    all_snapshots = []
    current_season = None
    
    # Trackers for position
    points = {}
    goal_diff = {}
    goals_for = {}
    team_divs = {} # Latest div for each team this season
    
    print("Running final pass and taking snapshots...")
    for idx, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        result = row['FTR']
        match_date = row['Date']
        season = row['Season']
        div = row['Div']
        
        # Season Snapshot (End of previous season)
        if current_season is not None and season != current_season:
            ranks = calculate_ranks(points, goal_diff, goals_for, team_divs)
            all_snapshots.extend(get_snapshot(system, target_teams, current_season, ranks))
            # Reset trackers for new season
            points, goal_diff, goals_for, team_divs = {}, {}, {}, {}

        current_season = season
        team_divs[home] = div
        team_divs[away] = div

        # --- UPDATE MODELS ---
        if result == 'H': score, p_h, p_a = 1.0, 3, 0
        elif result == 'A': score, p_h, p_a = 0.0, 0, 3
        else: score, p_h, p_a = 0.5, 1, 1
        
        fthg, ftag = row['FTHG'], row['FTAG']
        points[home] = points.get(home, 0) + p_h
        points[away] = points.get(away, 0) + p_a
        goal_diff[home] = goal_diff.get(home, 0) + (fthg - ftag)
        goal_diff[away] = goal_diff.get(away, 0) + (ftag - fthg)
        goals_for[home] = goals_for.get(home, 0) + fthg
        goals_for[away] = goals_for.get(away, 0) + ftag
        
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
        
    # Final Snapshot
    ranks = calculate_ranks(points, goal_diff, goals_for, team_divs)
    all_snapshots.extend(get_snapshot(system, target_teams, current_season, ranks))
        
    return all_snapshots

if __name__ == "__main__":
    teams = ["Colchester", "Southend", "Yeovil"]
    snapshots = extract_team_history(teams)
    res_df = pd.DataFrame(snapshots)
    res_df.to_csv("data/teams_history.csv", index=False)
    print("History saved to data/teams_history.csv")
