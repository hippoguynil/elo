import pandas as pd
import matplotlib.pyplot as plt

def plot_history(csv_path, team_name, output_path):
    df = pd.read_csv(csv_path)
    # Filter for the specific team
    team_df = df[df['Team'] == team_name].copy()
    
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # --- Axis 1: Model Ratings ---
    color_elo = 'tab:blue'
    color_glicko = 'tab:red'
    color_ts = 'tab:green'
    
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('Model Ratings', fontsize=12)
    
    ax1.plot(team_df['Season'].astype(str), team_df['Elo'], label='Elo', color=color_elo, linewidth=2)
    ax1.plot(team_df['Season'].astype(str), team_df['Glicko'], label='Glicko-2', color=color_glicko, linestyle='--', alpha=0.8)
    ax1.plot(team_df['Season'].astype(str), team_df['TrueSkill'], label='TrueSkill', color=color_ts, linestyle=':', alpha=0.8)
    
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.2)
    
    # --- Axis 2: League Position (Inverted) ---
    ax2 = ax1.twinx()
    color_pos = 'black'
    ax2.set_ylabel('League Position (Global 1-92)', color=color_pos, fontsize=12)
    
    ax2.plot(team_df['Season'].astype(str), team_df['Pos'], label='League Position', color=color_pos, linewidth=3, alpha=0.3)
    ax2.set_ylim(92, 1) # Invert axis: 1 at top, 92 at bottom
    ax2.tick_params(axis='y', labelcolor=color_pos)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title(f"{team_name} - Ratings vs. League Position (1993-2025)", fontsize=16)
    plt.xticks(rotation=45)
    
    # Reduce x-axis labels if too many
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Enhanced plot saved to {output_path}")

if __name__ == "__main__":
    plot_history("data/teams_history.csv", "Colchester", "colchester_enhanced.png")
