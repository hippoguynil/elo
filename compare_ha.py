
from football_models import FootballSystem

# Calibrated Params (from walkthrough)
ELO_OPT = {'k': 18, 'home_advantage': 54}
GLICKO_OPT = {'tau': 1.12, 'home_advantage': 10.1}
TS_OPT = {'beta': 5.26, 'home_advantage': 0.03}

print("--- Running WITH Home Advantage ---")
fs_with = FootballSystem(
    elo_config=ELO_OPT,
    glicko_config=GLICKO_OPT,
    ts_config=TS_OPT
)
loss_e_1, loss_g_1, loss_t_1 = fs_with.run_training(silent=True)
print(f"Elo: {loss_e_1:.4f}, Glicko: {loss_g_1:.4f}, TrueSkill: {loss_t_1:.4f}")

print("\n--- Running WITHOUT Home Advantage ---")
fs_without = FootballSystem(
    elo_config={**ELO_OPT, 'home_advantage': 0},
    glicko_config={**GLICKO_OPT, 'home_advantage': 0},
    ts_config={**TS_OPT, 'home_advantage': 0}
)
loss_e_0, loss_g_0, loss_t_0 = fs_without.run_training(silent=True)
print(f"Elo: {loss_e_0:.4f}, Glicko: {loss_g_0:.4f}, TrueSkill: {loss_t_0:.4f}")


print("\n--- Running with NEGATIVE Home Advantage (Bias Correction) ---")
# Testing a moderate negative bias
fs_neg = FootballSystem(
    elo_config={'k': 18, 'home_advantage': -50},
    glicko_config={'tau': 1.12, 'home_advantage': -10},
    ts_config={'beta': 5.26, 'home_advantage': -1.0}
)
loss_e_n, loss_g_n, loss_t_n = fs_neg.run_training(silent=True)
print(f"Elo: {loss_e_n:.4f}, Glicko: {loss_g_n:.4f}, TrueSkill: {loss_t_n:.4f}")

