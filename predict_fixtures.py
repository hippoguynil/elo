import pandas as pd
from datetime import datetime
from football_models import FootballSystem

def predict_group(system, title, fixtures):
    print(f"\n{title}")
    print("=" * 110)
    print(f"{'Match':<35} | {'Elo (H%)':<10} | {'Glicko (H%)':<10} | {'TrueSkill (H%)':<12} | {'Poisson (H%)':<10}")
    print("-" * 110)
    for home, away in fixtures:
        preds = system.predict_match(home, away)
        elo_p = preds['elo']['prob_home_win']
        glicko_p = preds['glicko']['prob_home_win']
        ts_p = preds['trueskill']['prob_home_win']
        poisson_p = preds['poisson']['prob_home_win']
        print(f"{home + ' vs ' + away:<35} | {elo_p:>9.1%} | {glicko_p:>9.1%} | {ts_p:>11.1%} | {poisson_p:>9.1%}")
    print("=" * 110)

def main():
    print("Initializing Football System...")
    system = FootballSystem()
    
    print("Calibrating models (Iterative Back-casting)...")
    system.calibrate(iterations=2, silent=True)
    
    # --- BOXING DAY (Friday, Dec 26) ---
    pl_boxing = [("Man United", "Newcastle")]
    
    champ_boxing = [
        ("Birmingham", "Derby"),
        ("Millwall", "Ipswich"),
        ("Coventry", "Swansea"),
        ("Leicester", "Watford"),
        ("Middlesbrough", "Blackburn"),
        ("Norwich", "Charlton"),
        ("Oxford", "Southampton"),
        ("Portsmouth", "QPR"),
        ("Sheffield Weds", "Hull"),
        ("Stoke", "Preston"),
        ("West Brom", "Bristol City"),
        ("Wrexham", "Sheffield United")
    ]
    
    l1_boxing = [
        ("Peterboro", "Leyton Orient"),
        ("AFC Wimbledon", "Stevenage"),
        ("Barnsley", "Mansfield"),
        ("Blackpool", "Doncaster"),
        ("Bolton", "Rotherham"),
        ("Bradford", "Wigan"),
        ("Burton", "Northampton"),
        ("Cardiff", "Exeter"), # Note: Cardiff in E2 in this data
        ("Huddersfield", "Port Vale"),
        ("Luton", "Wycombe"),
        ("Plymouth", "Reading"),
        ("Stockport", "Lincoln")
    ]

    l2_boxing = [
         ("Accrington", "Barrow"),
         ("Bristol Rvs", "Bromley"),
         ("Cheltenham", "Shrewsbury"),
         ("Chesterfield", "Notts County"),
         ("Crawley Town", "Colchester"),
         ("Gillingham", "Cambridge"),
         ("Grimsby", "Oldham"),
         ("MK Dons", "Swindon"),
         ("Newport County", "Barnet"),
         ("Salford", "Harrogate"),
         ("Tranmere", "Fleetwood Town"),
         ("Walsall", "Crewe")
    ]

    # --- SATURDAY (Dec 27) ---
    pl_sat = [
        ("Nott'm Forest", "Man City"),
        ("Arsenal", "Brighton"),
        ("Brentford", "Bournemouth"),
        ("Burnley", "Everton"),
        ("Liverpool", "Wolves"),
        ("West Ham", "Fulham"),
        ("Chelsea", "Aston Villa")
    ]

    # --- SUNDAY (Dec 28) ---
    pl_sun = [
        ("Sunderland", "Leeds"),
        ("Crystal Palace", "Tottenham")
    ]

    predict_group(system, "FRIDAY DEC 26 (BOXING DAY) - PREMIER LEAGUE", pl_boxing)
    predict_group(system, "FRIDAY DEC 26 (BOXING DAY) - CHAMPIONSHIP", champ_boxing)
    predict_group(system, "FRIDAY DEC 26 (BOXING DAY) - LEAGUE ONE", l1_boxing)
    predict_group(system, "FRIDAY DEC 26 (BOXING DAY) - LEAGUE TWO", l2_boxing)
    predict_group(system, "SATURDAY DEC 27 - PREMIER LEAGUE", pl_sat)
    predict_group(system, "SUNDAY DEC 28 - PREMIER LEAGUE", pl_sun)

if __name__ == "__main__":
    main()
