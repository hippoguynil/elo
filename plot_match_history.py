import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_match_history(csv_path, team_name, output_path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot Ratings
    ax.plot(df['Date'], df['Elo'], label='Elo', color='blue', linewidth=2)
    ax.plot(df['Date'], df['Glicko'], label='Glicko-2', color='red', linestyle='--', alpha=0.7)
    ax.plot(df['Date'], df['TrueSkill'], label='TrueSkill', color='green', linestyle=':', alpha=0.7)
    
    # Event Marker: Cowley Appointment (Jan 4, 2024)
    cowley_date = pd.to_datetime("2024-01-04")
    ax.axvline(x=cowley_date, color='black', linestyle='-', linewidth=2, alpha=0.8)
    ax.text(cowley_date, ax.get_ylim()[1], ' Cowleys Appointed (Jan 2024)', rotation=0, verticalalignment='bottom', fontweight='bold')
    
    # Season Markers (approx Aug 1st)
    seasons = ["2022-08-01", "2023-08-01", "2024-08-01", "2025-08-01"]
    for s_date in seasons:
        dt = pd.to_datetime(s_date)
        if dt > df['Date'].min() and dt < df['Date'].max():
            ax.axvline(x=dt, color='gray', linestyle='--', linewidth=1, alpha=0.4)
            ax.text(dt, ax.get_ylim()[0], f" {dt.year}/{dt.year+1-2000} Season", alpha=0.6, verticalalignment='bottom')

    # Formatting
    ax.set_title(f"{team_name} Match-by-Match Ratings (2021-2025)", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Rating", fontsize=12)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Match-by-Match plot saved to {output_path}")

if __name__ == "__main__":
    plot_match_history("data/colchester_match_history.csv", "Colchester", "colchester_granular_history.png")
