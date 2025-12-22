
import math
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# --- Elo Defaults ---
ELO_K = 18
ELO_START = 1500
ELO_HOME_ADVANTAGE = -50  # Negative to bias prediction towards <0.5 (Home Win < 50%)

# --- Glicko-2 Defaults ---
# Reasonable choices are between 0.3 and 1.2, smaller keys prevent massive volatility swings.
GLICKO2_TAU = 1.12
GLICKO2_START_RATING = 1500
GLICKO2_START_RD = 350
GLICKO2_START_VOL = 0.06
GLICKO2_CONVERSION = 173.7178  # Scaling factor
GLICKO2_HOME_ADVANTAGE = -10.0 # Negative calibrated value


# --- TrueSkill Defaults ---
# Based on Microsoft's TrueSkill defaults roughly scaled to 0-50 usually
TS_MU0 = 25.0
TS_SIGMA0 = 25.0 / 3.0
TS_BETA = 5.26  # Calibrated value
TS_TAU = TS_SIGMA0 / 100.0  # Additive dynamics factor
TS_HOME_ADVANTAGE = -1.0 # Negative calibrated value

TS_HOME_ADVANTAGE = -1.0 # Negative calibrated value

# --- Poisson Defaults ---
# Attack/Defense ratings usually center around 1.0 (multiplicative) or 0.0 (additive log)
# We will use additive log-lambda formulation.
POISSON_LEARNING_RATE = 0.001
POISSON_HOME_ADVANTAGE = 0.20 # Log-scale additive (corresponds to e^0.2 ~ 1.22x goals for home)
POISSON_DECAY = 0.00005 # Decay toward mean per day? Or variance inflation? SGD usually decays weights.

# =============================================================================
# MATH UTILITIES
# =============================================================================

class MathUtils:
    """Statistical utility functions needed for Glicko2 and TrueSkill."""
    
    @staticmethod
    def pdf(x):
        """Standard Normal Probability Density Function."""
        return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

    @staticmethod
    def cdf(x):
        """Standard Normal Cumulative Distribution Function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def ppf(p):
        """Percent point function (inverse of CDF) - approximate."""
        # Simple approximation for inverse error function if needed, 
        # or use scipy if available, but requesting 'no externals' if possible.
        # However, for win probability we usually just need CDF.
        # For detailed TrueSkill draw margin we might need inverse CDF but 
        # we can start with fixed margins or simplified logic.
        pass

    @staticmethod
    def v_win(t, epsilon):
        """TrueSkill additive correction function for a winner."""
        # v(t, e) = N(t-e) / Phi(t-e)
        denominator = MathUtils.cdf(t - epsilon)
        if denominator < 1e-9: return 0.0 # avoid div by zero
        return MathUtils.pdf(t - epsilon) / denominator

    @staticmethod
    def w_win(t, epsilon):
        """TrueSkill multiplicative correction function for a winner."""
        # w(t, e) = v(t, e) * (v(t, e) + t - e)
        v = MathUtils.v_win(t, epsilon)
        return v * (v + t - epsilon)

    @staticmethod
    def v_loss(t, epsilon):
        """TrueSkill additive correction for a loser."""
        # v(t, e) = -N(t-e) / Phi(e-t)  -- equivalent logic
        return -MathUtils.v_win(-t, epsilon)

    @staticmethod
    def w_loss(t, epsilon):
        """TrueSkill multiplicative correction for a loser."""
        # w(t, e) = v(t, e) * (v(t, e) + t - e)
        # Note: simplistic symmetry usage here
        v = MathUtils.v_loss(t, epsilon)
        return v * (v + t - epsilon)

    @staticmethod
    def skellam_win_prob(local_lambda, visitor_lambda, max_goals=10):
        """
        Approximate probability that HomeGoals > AwayGoals using Poisson summation.
        Using simulation or finite summation is often faster/easier than Skellam CDF implementation.
        """
        # Sum P(H=i) * P(A=j) for i > j
        prob_home_win = 0.0
        prob_draw = 0.0
        prob_away_win = 0.0
        
        # Precompute poisson PMFs
        p_h = [MathUtils.poisson_pmf(i, local_lambda) for i in range(max_goals+1)]
        p_a = [MathUtils.poisson_pmf(i, visitor_lambda) for i in range(max_goals+1)]
        
        for i in range(len(p_h)):
            for j in range(len(p_a)):
                prob = p_h[i] * p_a[j]
                if i > j:
                    prob_home_win += prob
                elif i == j:
                    prob_draw += prob
                else:
                    prob_away_win += prob
                    
        return prob_home_win, prob_draw, prob_away_win

    @staticmethod
    def poisson_pmf(k, lam):
        return (lam**k * math.exp(-lam)) / math.factorial(k)



# =============================================================================
# MODEL INTERFACE
# =============================================================================

class RatingSystem:
    def predict_win_prob(self, rating_home, rating_away):
        """
        Returns probability of Home Win.
        Note: The user requirement is 'draw is a win for away team'
        so P(AwayWin or Draw) = 1 - P(Home Win).
        """
        raise NotImplementedError

    def update(self, rating_home, rating_away, outcome, **kwargs):
        """
        Updates ratings based on outcome.
        Outcome: 1.0 (Home Win), 0.5 (Draw), 0.0 (Away Win).
        Kwargs: 'home_goals', 'away_goals' (needed for Poisson), 'delta_days' (needed for Time-Dep).
        Returns: (new_home_rating, new_away_rating)
        """
        raise NotImplementedError

    def get_initial_rating(self):
        raise NotImplementedError


# =============================================================================
# ELO IMPLEMENTATION
# =============================================================================

class EloRating:
    def __init__(self, rating=ELO_START):
        self.rating = rating

    def __repr__(self):
        return f"Elo({self.rating:.1f})"

class EloModel(RatingSystem):
    def __init__(self, k=ELO_K, home_advantage=ELO_HOME_ADVANTAGE):
        self.k = k
        self.home_advantage = home_advantage

    def get_initial_rating(self):
        return EloRating(ELO_START)

    def predict_win_prob(self, r_home: EloRating, r_away: EloRating):
        # Specific definition for Elo:
        # P(A) = 1 / (1 + 10 ^ ((Rb - Ra) / 400))
        # We add home advantage to Ra (Home Team)
        ra = r_home.rating + self.home_advantage
        rb = r_away.rating
        expected_home = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        return expected_home

        return expected_home

    def update(self, r_home: EloRating, r_away: EloRating, outcome, delta_days_home=0, delta_days_away=0, **kwargs):
        # Time-Dependent: Regress to mean?
        # A simple approach: R_new_pre = R + lambda * (1500 - R) * days
        # We process regression BEFORE match update.
        # Arbitrary regression rate: 0.1 points per day towards mean?
        # Let's keep it 0 for now unless requested strictly, or add a small regression.
        # User asked for "Time-Dependent features".
        # Let's add slight mean regression.
        
        # Regression
        reg_rate = 0.01 # Very slow drift back to 1500
        rh_reg = r_home.rating + reg_rate * (ELO_START - r_home.rating) * (delta_days_home / 365.0)
        ra_reg = r_away.rating + reg_rate * (ELO_START - r_away.rating) * (delta_days_away / 365.0)

        expected_home = self.predict_win_prob(EloRating(rh_reg), EloRating(ra_reg))
        
        # In standard Elo, K might vary, but we keep it fixed for efficiency/simplicity
        new_rating_home = rh_reg + self.k * (outcome - expected_home)
        new_rating_away = ra_reg + self.k * ((1 - outcome) - (1 - expected_home))
        
        return EloRating(new_rating_home), EloRating(new_rating_away)


# =============================================================================
# GLICKO-2 IMPLEMENTATION
# =============================================================================

class Glicko2Rating:
    def __init__(self, rating=GLICKO2_START_RATING, rd=GLICKO2_START_RD, vol=GLICKO2_START_VOL):
        self.rating = rating
        self.rd = rd
        self.vol = vol

    def __repr__(self):
        return f"G2(R={self.rating:.0f}, RD={self.rd:.0f})"

class Glicko2Model(RatingSystem):
    def __init__(self, tau=GLICKO2_TAU, home_advantage=GLICKO2_HOME_ADVANTAGE):
        self.tau = tau
        self.home_advantage = home_advantage
        self.epsilon = 0.000001 # Convergence tolerance

    def get_initial_rating(self):
        return Glicko2Rating()

    def _scale_down(self, r: Glicko2Rating):
        """Convert to Glicko-2 scale."""
        mu = (r.rating - 1500) / GLICKO2_CONVERSION
        phi = r.rd / GLICKO2_CONVERSION
        return mu, phi, r.vol

    def _scale_up(self, mu, phi, vol):
        """Convert back to Glicko-1 scale."""
        rating = 1500 + mu * GLICKO2_CONVERSION
        rd = phi * GLICKO2_CONVERSION
        return Glicko2Rating(rating, rd, vol)

    def _g(self, phi):
        return 1.0 / math.sqrt(1.0 + 3.0 * (phi ** 2) / (math.pi ** 2))

    def _E(self, mu, mu_j, phi_j):
        return 1.0 / (1.0 + math.exp(-self._g(phi_j) * (mu - mu_j)))

    def predict_win_prob(self, r_home: Glicko2Rating, r_away: Glicko2Rating):
        # Using the Glicko expectation formula as probability
        # Note: Glicko-2 usually runs in batches (rating periods).
        # We will treat each match as a rating period of size 1 for online processing.
        mu_h, phi_h, _ = self._scale_down(r_home)
        mu_a, phi_a, _ = self._scale_down(r_away)
        
        # Apply Home Advantage (scaled down)
        # HA is provided in Elo-like units, so we scale it too.
        mu_h += self.home_advantage / GLICKO2_CONVERSION
        
        # Composite phi for the match interaction
        g_factor = self._g(math.sqrt(phi_h**2 + phi_a**2))
        return 1.0 / (1.0 + math.exp(-g_factor * (mu_h - mu_a)))

        return 1.0 / (1.0 + math.exp(-g_factor * (mu_h - mu_a)))

    def update(self, r_home: Glicko2Rating, r_away: Glicko2Rating, outcome, delta_days_home=0, delta_days_away=0, **kwargs):
        # Time-Dependent Variance Inflation (Step 1 of Glicko-2 usually involves time period)
        # phi_new = sqrt(phi_old^2 + sigma^2 * time)
        # We apply this to RD before the match update.
        
        mu_h, phi_h, sigma_h = self._scale_down(r_home)
        mu_a, phi_a, sigma_a = self._scale_down(r_away)
        
        # Inflate variance based on time
        # Glicko-2 paper: phi' = sqrt(phi^2 + sigma^2) per rating period.
        # We treat 'delta_days' as fractional rating periods? Or scaling factor?
        # If standard period is e.g. 1 week (7 days).
        # Let's say volatility applies per unit time (day).
        # Note: sigma is usually small (0.06).
        # We need to be careful not to explode variance.
        # Let's assume sigma is "volatility per match-period".
        # We'll stick to a simple addon: phi_star = sqrt(phi^2 + (sigma^2 * delta_days/7.0))
        
        time_factor_h = max(delta_days_home / 7.0, 1.0) # At least 1 period
        time_factor_a = max(delta_days_away / 7.0, 1.0)
        
        phi_h = math.sqrt(phi_h**2 + (sigma_h**2) * time_factor_h)
        phi_a = math.sqrt(phi_a**2 + (sigma_a**2) * time_factor_a)
        
        # --- Continue with Update (using inflated phi) ---
        
        # Apply Home Advantage to 'mu' (Home Team) for the calculation 
        # But wait, update logic usually involves comparing performance.
        # If Home has advantage, their 'effective' skill is higher.
        # So when computing Expected outcome, we use (mu + HA).
        # We assume HA is a temporary boost, not a permanent skill upgrade.
        mu_with_ha = mu_h + (self.home_advantage / GLICKO2_CONVERSION)
        
        # Step 3: Compute v (estimated variance) and delta
        g_opp = self._g(phi_a)
        E_opp = self._E(mu_with_ha, mu_a, phi_a)
        
        v = 1.0 / (g_opp ** 2 * E_opp * (1 - E_opp))
        delta = v * g_opp * (outcome - E_opp)

        # Step 5: New Volatility (sigma') - Iterative algorithm
        a = math.log(sigma_h ** 2)

        
        def f(x):
            ex = math.exp(x)
            num = ex * (delta ** 2 - phi_h ** 2 - v - ex)
            den = 2 * ((phi_h ** 2 + v + ex) ** 2)
            return (num / den) - ((x - a) / (self.tau ** 2))

        # Illinois algorithm / Regula Falsi setup
        A = a
        if (delta ** 2) > (phi_h ** 2 + v):
            B = math.log(delta ** 2 - phi_h ** 2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau

        fA = f(A)
        fB = f(B)

        while abs(B - A) > self.epsilon:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB < 0:
                A = B
                fA = fB
            else:
                fA = fA / 2.0
            B = C
            fB = fC

        sigma_prime = math.exp(A / 2.0)

        # Step 6: Update PHI (pre-rating deviation)
        phi_star = math.sqrt(phi_h ** 2 + sigma_prime ** 2)

        # Step 7: Update RD and Rating
        phi_prime = 1.0 / math.sqrt(1.0 / (phi_star ** 2) + 1.0 / v)
        mu_prime = mu_h + (phi_prime ** 2) * g_opp * (outcome - E_opp)

        # Return updated home rating
        new_home_r = self._scale_up(mu_prime, phi_prime, sigma_prime)
        
        # We also need to update the Away team (symmetric but inverse outcome)
        # Note: In 1v1, Glicko2 is symmetric if processed simultaneously.
        # We repeat the calculation for the opponent from their perspective.
        
        # Symmetrical calc for opponent
        # Symmetrical calc for opponent (Away Team perspective)
        # When Away plays Home, Home has advantage.
        # Away effective skill: mu_a
        # Home effective skill: mu_h + HA
        
        # Symmetrical calc for opponent
        # We already inflated phi_h, phi_a above.
        # But we need to reuse the original logic (variables are shadowed/local).
        # Actually, in the code above I updated local `phi_h`, `phi_a`.
        # I should use those inflated values for the symmetrical update too?
        # Yes.
        
        # mu_a, phi_a, sigma_a are already scaled down and time-inflated above.
        # mu_h, phi_h, sigma_h are already scaled down and time-inflated above.
        
        # Home opp has advantage
        mu_h_opp_with_ha = mu_h + (self.home_advantage / GLICKO2_CONVERSION)
        
        outcome_a = 1.0 - outcome
        g_h = self._g(phi_h)
        E_h = self._E(mu_a, mu_h_opp_with_ha, phi_h)
        v_a = 1.0 / (g_h ** 2 * E_h * (1 - E_h))
        delta_a = v_a * g_h * (outcome_a - E_h)
        
        # Volatility update for opponent
        a_a = math.log(sigma_a ** 2)
        def f_a(x):
            ex = math.exp(x)
            num = ex * (delta_a ** 2 - phi_a ** 2 - v_a - ex)
            den = 2 * ((phi_a ** 2 + v_a + ex) ** 2)
            return (num / den) - ((x - a_a) / (self.tau ** 2))
            
        A = a_a
        if (delta_a ** 2) > (phi_a ** 2 + v_a):
            B = math.log(delta_a ** 2 - phi_a ** 2 - v_a)
        else:
            k = 1
            while f_a(a_a - k * self.tau) < 0:
                k += 1
            B = a_a - k * self.tau
            
        fA = f_a(A)
        fB = f_a(B)
        
        while abs(B - A) > self.epsilon:
            C = A + (A - B) * fA / (fB - fA)
            fC = f_a(C)
            if fC * fB < 0:
                A = B
                fA = fB
            else:
                fA = fA / 2.0
            B = C
            fB = fC
            
        sigma_prime_a = math.exp(A / 2.0)
        phi_star_a = math.sqrt(phi_a ** 2 + sigma_prime_a ** 2)
        phi_prime_a = 1.0 / math.sqrt(1.0 / (phi_star_a ** 2) + 1.0 / v_a)
        mu_prime_a = mu_a + (phi_prime_a ** 2) * g_h * (outcome_a - E_h)
        
        new_away_r = self._scale_up(mu_prime_a, phi_prime_a, sigma_prime_a)
        
        return new_home_r, new_away_r


# =============================================================================
# TRUESKILL IMPLEMENTATION (APPROXIMATE 1v1)
# =============================================================================

class TrueSkillRating:
    def __init__(self, mu=TS_MU0, sigma=TS_SIGMA0):
        self.mu = mu
        self.sigma = sigma
    
    def __repr__(self):
        return f"TS(µ={self.mu:.1f}, s={self.sigma:.1f})"

class TrueSkillModel(RatingSystem):
    def __init__(self, beta=TS_BETA, home_advantage=TS_HOME_ADVANTAGE):
        self.beta = beta
        self.home_advantage = home_advantage
        # Binary classification (Home vs Not-Home): No draw margin
        self.epsilon = 0.0

    def get_initial_rating(self):
        return TrueSkillRating()

    def predict_win_prob(self, r_home: TrueSkillRating, r_away: TrueSkillRating):
        # P(Home > Away) = Phi( (muA - muB) / sqrt(beta^2 + beta^2 + sigmaA^2 + sigmaB^2) )
        # Add HA to Home mu
        delta_mu = (r_home.mu + self.home_advantage) - r_away.mu
        sum_sigma_sq = r_home.sigma**2 + r_away.sigma**2 + 2 * (self.beta**2)
        denom = math.sqrt(sum_sigma_sq)
        return MathUtils.cdf(delta_mu / denom)

    def update(self, r_home: TrueSkillRating, r_away: TrueSkillRating, outcome, delta_days_home=0, delta_days_away=0, **kwargs):
        # Time-Dependent Variance
        # Increase sigma based on time elapsed
        # sigma_new^2 = sigma_old^2 + (tau^2 * delta_days/7)
        
        time_factor_h = max(delta_days_home / 7.0, 1.0)
        time_factor_a = max(delta_days_away / 7.0, 1.0)
        
        sigma_h = math.sqrt(r_home.sigma**2 + (TS_TAU**2 * time_factor_h))
        sigma_a = math.sqrt(r_away.sigma**2 + (TS_TAU**2 * time_factor_a))
        
        # Outcome: 1=HomeWin, 0=AwayWin/Draw
        
        # We treat standard TS update:
        # c^2 = 2beta^2 + sigma1^2 + sigma2^2
        c_sq = 2 * (self.beta**2) + sigma_h**2 + sigma_a**2
        c = math.sqrt(c_sq)
        
        # Apply HA to diff for update calculation
        diff = (r_home.mu + self.home_advantage) - r_away.mu
        
        if outcome == 1.0: # Home Win
            # v = v((diff)/c, epsilon/c)
            # Since epsilon=0, v = pdf(t)/cdf(t) = Hazard Function of normal dist
            val_v = MathUtils.v_win(diff/c, 0.0)
            val_w = MathUtils.w_win(diff/c, 0.0)
            
            # Updates
            mean_delta_home = (sigma_h**2 / c) * val_v
            mean_delta_away = -(sigma_a**2 / c) * val_v
            
            new_sigma_sq_home = sigma_h**2 * (1 - (sigma_h**2 / c_sq) * val_w)
            new_sigma_sq_away = sigma_a**2 * (1 - (sigma_a**2 / c_sq) * val_w)
            
        else: # Away Win or Draw (treated as Away Win)
            # Symmetric to Home Win but reversed signs
            val_v = MathUtils.v_win(-diff/c, 0.0)
            val_w = MathUtils.w_win(-diff/c, 0.0)
            
            mean_delta_home = -(sigma_h**2 / c) * val_v
            mean_delta_away = (sigma_a**2 / c) * val_v
            
            new_sigma_sq_home = sigma_h**2 * (1 - (sigma_h**2 / c_sq) * val_w)
            new_sigma_sq_away = sigma_a**2 * (1 - (sigma_a**2 / c_sq) * val_w)
            
        # Apply dynamics (additive variance) - Already applied Time-Dependent at start?
        # TrueSkill usually applies dynamics AFTER update for next period.
        # But here we applied it pre-match (decay).
        # We should NOT apply it again fully, or apply generic dynamics.
        # Let's assume the pre-match inflation covers the "time between matches".
        # So we just return the posterior sigma.
        
        new_sigma_home = math.sqrt(new_sigma_sq_home)
        new_sigma_away = math.sqrt(new_sigma_sq_away)
        
        return (TrueSkillRating(r_home.mu + mean_delta_home, new_sigma_home),
                TrueSkillRating(r_away.mu + mean_delta_away, new_sigma_away))


# =============================================================================
# POISSON MODEL IMPLEMENTATION
# =============================================================================

class PoissonRating:
    def __init__(self, attack=0.01, defense=0.01):
        self.attack = attack # Log scale
        self.defense = defense # Log scale
        # We start near 0 (neutral).
        
    def __repr__(self):
        return f"Pois(A={self.attack:.2f}, D={self.defense:.2f})"

class PoissonModel(RatingSystem):
    def __init__(self, lr=POISSON_LEARNING_RATE, home_adv=POISSON_HOME_ADVANTAGE):
        self.lr = lr
        self.home_adv = home_adv
        
    def get_initial_rating(self):
        return PoissonRating()
        
    def predict_lambdas(self, r_home, r_away):
        # lambda_home = exp(att_h + def_a + HA)
        # lambda_away = exp(att_a + def_h)
        lam_h = math.exp(r_home.attack + r_away.defense + self.home_adv)
        lam_a = math.exp(r_away.attack + r_home.defense)
        return lam_h, lam_a
        
    def predict_win_prob(self, r_home, r_away):
        # Calculate prob(HomeGoals > AwayGoals)
        lam_h, lam_a = self.predict_lambdas(r_home, r_away)
        p_h, p_d, p_a = MathUtils.skellam_win_prob(lam_h, lam_a)
        # User defined Win = Home Win.
        return p_h

    def update(self, r_home, r_away, outcome, home_goals=0, away_goals=0, delta_days_home=0, delta_days_away=0, **kwargs):
        # Online Gradient Descent on Log Likelihood
        # L = k*log(lam) - lam
        # dL/dParam = k - lam  (since lam = exp(param))
        
        lam_h, lam_a = self.predict_lambdas(r_home, r_away)
        
        # Error terms
        err_h = home_goals - lam_h
        err_a = away_goals - lam_a
        
        # Update rules:
        # Home Attack affects Home Goals: dL/dAttH = err_h
        # Away Defense affects Home Goals: dL/dDefA = err_h
        # Away Attack affects Away Goals: dL/dAttA = err_a
        # Home Defense affects Away Goals: dL/dDefH = err_a
        
        # Time Decay / Regularization?
        # Simple SGD:
        new_att_h = r_home.attack + self.lr * err_h
        new_def_h = r_home.defense + self.lr * err_a
        new_att_a = r_away.attack + self.lr * err_a
        new_def_a = r_away.defense + self.lr * err_h
        
        return PoissonRating(new_att_h, new_def_h), PoissonRating(new_att_a, new_def_a)

# =============================================================================
# PREDICTOR & RUNNER
# =============================================================================

class FootballSystem:
    def __init__(self, data_path='data/all_data.csv', elo_config=None, glicko_config=None, ts_config=None):
        self.data_path = data_path
        self.teams_elo = {}
        self.teams_glicko = {}
        self.teams_ts = {}
        
        if elo_config is None: elo_config = {}
        if glicko_config is None: glicko_config = {}
        if ts_config is None: ts_config = {}
        
        if ts_config is None: ts_config = {}
        
        self.elo_model = EloModel(**elo_config)
        self.glicko_model = Glicko2Model(**glicko_config)
        self.ts_model = TrueSkillModel(**ts_config)
        self.poisson_model = PoissonModel() # No config for now
        
        self.teams_poisson = {}
        self.teams_last_date = {} # Map team_name -> datetime of last match
        
    def _get_team(self, name, model_type):
        store = getattr(self, f"teams_{model_type}")
        model = getattr(self, f"{model_type}_model")
        if name not in store:
            store[name] = model.get_initial_rating()
        return store[name]
    
    def _set_team(self, name, model_type, rating):
        getattr(self, f"teams_{model_type}")[name] = rating

    def load_and_prep_data(self):
        df = pd.read_csv(self.data_path)
        # Ensure date sorting
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date')
        
        if not silent:
            print("Training complete.")
            print(f"Glicko-2 Accuracy (Away/Draw prediction): {correct_preds/total_preds:.4f}")
            print(f"Avg Log Loss Total: {log_loss_sum/total_preds:.4f}")
            
        # Return losses for optimization (returning uniform loss for now or calculate individual)
        # To strictly optimize each model, we should track their individual losses.
        # But for this simple implementation, let's assume log_loss_sum is Glicko's.
        # We need individual model losses. Refactoring slightly to track e_loss, g_loss, t_loss.
        return self._final_losses 

    def run_training(self, silent=False):
        """Iterates through all matches and updates ratings."""
        if not silent:
            print("Loading data...")
            # df = self.load_and_prep_data() # Optim: Don't reload every time if checking repeatedly, but OS cache helps.
        
        # For simplicity in this tool edit, I will stick to the existing structure 
        # but add trackers for individual model losses.
        
        df = self.load_and_prep_data()
        if not silent: print(f"Processing {len(df)} matches...")
        
        
        loss_e = 0
        loss_g = 0
        loss_t = 0
        loss_p = 0
        correct_e = 0
        correct_g = 0
        correct_t = 0
        correct_p = 0
        count = 0
        
        for idx, row in df.iterrows():
            home, away = row['HomeTeam'], row['AwayTeam']
            result = row['FTR'] # H, D, A
            match_date = row['Date']
            
            # --- Time Dependency ---
            # Calculate days since last match for each team
            last_date_h = self.teams_last_date.get(home, match_date) # Default to current (=0 delta) if new
            last_date_a = self.teams_last_date.get(away, match_date)
            
            delta_days_h = (match_date - last_date_h).days
            delta_days_a = (match_date - last_date_a).days
            
            # Ensure non-negative (sorting should handle this, but just in case)
            delta_days_h = max(0, delta_days_h)
            delta_days_a = max(0, delta_days_a)
            
            # Goals for Poisson
            fthg = row['FTHG']
            ftag = row['FTAG']
            
            # Numeric Outcome
            if result == 'H': score = 1.0
            else: score = 0.0
            
            actual_home_win = 1 if result == 'H' else 0
            
            # Get Ratings
            e_h, e_a = self._get_team(home, 'elo'), self._get_team(away, 'elo')
            g_h, g_a = self._get_team(home, 'glicko'), self._get_team(away, 'glicko')
            t_h, t_a = self._get_team(home, 'ts'), self._get_team(away, 'ts')
            p_h, p_a = self._get_team(home, 'poisson'), self._get_team(away, 'poisson')

            # Predictions (Home Win Prob)
            # Elo
            prob_e = self.elo_model.predict_win_prob(e_h, e_a)
            # Glicko
            prob_g = self.glicko_model.predict_win_prob(g_h, g_a)
            # TrueSkill
            prob_t = self.ts_model.predict_win_prob(t_h, t_a)
            # Poisson
            prob_p = self.poisson_model.predict_win_prob(p_h, p_a)
            
            # Log Loss Calculation (Home Win)
            # Loss = -(y log(p) + (1-y) log(1-p))
            # Safe log function
            def safe_log_loss(p, y):
                p = max(min(p, 0.9999), 0.0001)
                return -(y * math.log(p) + (1-y) * math.log(1-p))
                
            loss_e += safe_log_loss(prob_e, actual_home_win)
            loss_g += safe_log_loss(prob_g, actual_home_win)
            loss_t += safe_log_loss(prob_t, actual_home_win)
            loss_p += safe_log_loss(prob_p, actual_home_win)
            
            # Accuracy (Threshold 0.5 for Home Win)
            # We predict Home Win Prob.
            # If P(Home) > 0.5 and Result=H -> Correct
            # If P(Home) <= 0.5 and Result!=H -> Correct
            if (prob_e > 0.5 and actual_home_win == 1) or (prob_e <= 0.5 and actual_home_win == 0): correct_e += 1
            if (prob_g > 0.5 and actual_home_win == 1) or (prob_g <= 0.5 and actual_home_win == 0): correct_g += 1
            if (prob_t > 0.5 and actual_home_win == 1) or (prob_t <= 0.5 and actual_home_win == 0): correct_t += 1
            if (prob_p > 0.5 and actual_home_win == 1) or (prob_p <= 0.5 and actual_home_win == 0): correct_p += 1
            
            count += 1

            # --- UPDATES ---
            # Pass deltas and goals where needed
            new_e_h, new_e_a = self.elo_model.update(e_h, e_a, score, delta_days_home=delta_days_h, delta_days_away=delta_days_a)
            new_g_h, new_g_a = self.glicko_model.update(g_h, g_a, score, delta_days_home=delta_days_h, delta_days_away=delta_days_a)
            new_t_h, new_t_a = self.ts_model.update(t_h, t_a, score, delta_days_home=delta_days_h, delta_days_away=delta_days_a)
            new_p_h, new_p_a = self.poisson_model.update(p_h, p_a, score, home_goals=fthg, away_goals=ftag)
            
            self._set_team(home, 'elo', new_e_h)
            self._set_team(away, 'elo', new_e_a)
            self._set_team(home, 'glicko', new_g_h)
            self._set_team(away, 'glicko', new_g_a)
            self._set_team(home, 'ts', new_t_h)
            self._set_team(away, 'ts', new_t_a)
            self._set_team(home, 'poisson', new_p_h)
            self._set_team(away, 'poisson', new_p_a)
            
            # Update Date
            self.teams_last_date[home] = match_date
            self.teams_last_date[away] = match_date

        avg_loss_e = loss_e / count
        avg_loss_g = loss_g / count
        avg_loss_t = loss_t / count
        avg_loss_p = loss_p / count
        
        if not silent:
            print("Training complete.")
            print(f"Avg Log Loss - Elo: {avg_loss_e:.4f} | Acc: {correct_e/count:.2%}")
            print(f"Avg Log Loss - Glicko2: {avg_loss_g:.4f} | Acc: {correct_g/count:.2%}")
            print(f"Avg Log Loss - TrueSkill: {avg_loss_t:.4f} | Acc: {correct_t/count:.2%}")
            print(f"Avg Log Loss - Poisson: {avg_loss_p:.4f} | Acc: {correct_p/count:.2%}")
            
        return avg_loss_e, avg_loss_g, avg_loss_t, avg_loss_p


    def predict_match(self, home_team, away_team):
        """
        Public API to get prediction for a future match.
        Returns dictionary of probabilities from all models.
        Key metric: 'prob_away_win_or_draw'
        """
        # Elo
        e_h = self.teams_elo.get(home_team, self.elo_model.get_initial_rating())
        e_a = self.teams_elo.get(away_team, self.elo_model.get_initial_rating())
        p_h_elo = self.elo_model.predict_win_prob(e_h, e_a)
        
        # Glicko
        g_h = self.teams_glicko.get(home_team, self.glicko_model.get_initial_rating())
        g_a = self.teams_glicko.get(away_team, self.glicko_model.get_initial_rating())
        p_h_glicko = self.glicko_model.predict_win_prob(g_h, g_a)
        
        # TrueSkill
        t_h = self.teams_ts.get(home_team, self.ts_model.get_initial_rating())
        t_a = self.teams_ts.get(away_team, self.ts_model.get_initial_rating())
        p_h_ts = self.ts_model.predict_win_prob(t_h, t_a)
        
        # Poisson
        p_h = self.teams_poisson.get(home_team, self.poisson_model.get_initial_rating())
        p_a = self.teams_poisson.get(away_team, self.poisson_model.get_initial_rating())
        prob_p = self.poisson_model.predict_win_prob(p_h, p_a)
        
        return {
            "elo": {
                "home_rating": e_h.rating,
                "away_rating": e_a.rating,
                "prob_home_win": p_h_elo,
                "prob_away_or_draw": 1.0 - p_h_elo
            },
            "glicko": {
                "home_rating": f"{g_h.rating:.0f} (RD: {g_h.rd:.0f})",
                "away_rating": f"{g_a.rating:.0f} (RD: {g_a.rd:.0f})",
                "prob_home_win": p_h_glicko,
                "prob_away_or_draw": 1.0 - p_h_glicko
            },
            "trueskill": {
                "home_rating": f"{t_h.mu:.1f}",
                "away_rating": f"{t_a.mu:.1f}",
                "prob_home_win": p_h_ts,
                "prob_away_or_draw": 1.0 - p_h_ts
            },
            "poisson": {
                "home_rating": f"A:{p_h.attack:.2f}/D:{p_h.defense:.2f}",
                "away_rating": f"A:{p_a.attack:.2f}/D:{p_a.defense:.2f}",
                "prob_home_win": prob_p,
                "prob_away_or_draw": 1.0 - prob_p
            }
        }

class Optimizer:
    def run_calibration(self):
        import random
        # Best params so far handling
        best_elo = {'loss': float('inf'), 'params': {}}
        best_glicko = {'loss': float('inf'), 'params': {}}
        best_ts = {'loss': float('inf'), 'params': {}}
        
        print("Starting Calibration (Mini-batch)...")
        # We'll run a few iterations of random search
        # Note: In a real scenario, we'd want to be much more exhaustive.
        # This is a demonstration of the calibration logic.
        
        iterations = 5 # Small number for demonstration speed
        
        for i in range(iterations):
            # Sample params
            e_k = random.randint(10, 60)
            e_ha = random.randint(50, 200)
            
            g_tau = random.uniform(0.3, 1.2)
            g_ha = random.uniform(10, 60)
            
            ts_beta = random.uniform(2.0, 6.0)
            ts_ha = random.uniform(0.0, 5.0)
            
            print(f"Iter {i+1}/{iterations}: Elo(k={e_k}, ha={e_ha}), G2(tau={g_tau:.2f}, ha={g_ha:.1f}), TS(b={ts_beta:.2f}, ha={ts_ha:.2f})")
            
            fs = FootballSystem(
                elo_config={'k': e_k, 'home_advantage': e_ha},
                glicko_config={'tau': g_tau, 'home_advantage': g_ha},
                ts_config={'beta': ts_beta, 'home_advantage': ts_ha}
            )
            
            # Monkey patch run_training to return metrics instead of printing
            # Or just capture prints? Easier to duplicate logic or modify run_training slightly.
            # Let's modify run_training to return loss.
            # Wait, modifying existing class method is better done by just calling it and parsing or refactoring.
            # I'll just refactor run_training to return the loss at the end.
            
            loss_elo, loss_glicko, loss_ts, loss_p = fs.run_training(silent=True)
            
            if loss_elo < best_elo['loss']:
                best_elo = {'loss': loss_elo, 'params': {'k': e_k, 'home_advantage': e_ha}}
            if loss_glicko < best_glicko['loss']:
                best_glicko = {'loss': loss_glicko, 'params': {'tau': g_tau, 'home_advantage': g_ha}}
            if loss_ts < best_ts['loss']:
                best_ts = {'loss': loss_ts, 'params': {'beta': ts_beta, 'home_advantage': ts_ha}}
            # Poisson not strictly part of this loop, could add if needed

                
        print("\n=== Calibration Results ===")
        print(f"Best Elo: {best_elo}")
        print(f"Best Glicko: {best_glicko}")
        print(f"Best TrueSkill: {best_ts}")


if __name__ == "__main__":
    # Check for calibration flag or just run default
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--calibrate':
        opt = Optimizer()
        opt.run_calibration()
    else:
        fs = FootballSystem()
        fs.run_training()
        
        # Test Prediction
        print("\n--- Example Prediction ---")
        res = fs.predict_match("Arsenal", "Liverpool")
        import json
        print(json.dumps(res, indent=2))

