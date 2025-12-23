import pandas as pd
from football_models import FootballSystem, POISSON_LEARNING_RATE

def main():
    print(f"Checking Poisson Performance with LR={POISSON_LEARNING_RATE}")
    system = FootballSystem()
    system.run_training(silent=True)
    
    # Inspect Ratings for top teams vs bottom teams from 2024/2025 season implied knowledge
    # Let's just pick known strong/weak teams
    teams = ['Man City', 'Liverpool', 'Arsenal', 'Southampton', 'Sheffield United', 'Wrexham']
    
    print("\nPoisson Ratings:")
    for team in teams:
        if team in system.teams_poisson:
            r = system.teams_poisson[team]
            print(f"{team:<20}: Att={r.attack:.4f}, Def={r.defense:.4f}")
        else:
            print(f"{team:<20}: Not found")

if __name__ == "__main__":
    main()
