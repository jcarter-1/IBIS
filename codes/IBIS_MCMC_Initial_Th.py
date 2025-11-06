import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import fsolve, brentq
from scipy.stats import gaussian_kde, norm, uniform, lognorm
from scipy.special import expit, log_ndtr
from scipy.interpolate import interp1d
from tqdm import tqdm, tnrange, tqdm_notebook
import dill as pickle
import time
import os
from joblib import Parallel, delayed
import random
import warnings
warnings.filterwarnings("ignore")

import sys


class IBIS_MCMC:
    """
    IBIS
    ----
    Determine unique initial thorium correction for U-Th speleothem samples with stratigraphic constraints.
    
    Input (NEEDED)
    ----
    Prior - Thor_KDE (Prior constructed from thoth sampler)
    Age_Maximum - Bottom age of speleothem at 5𝜎
    Age_Uncertainties - Measured Age uncertainties
    data - Dataframe of required data (from model set up in IBIS_Main)
    sample_name - Defined in instance of IBIS Main
    
    (User Defined)
    n_chains - number of chains run (3 is the preset but run as many as you like)
    iterations - number of samples from posterior
    burn_in - number of tune up samples before sampling from posterior
    Start_from_Pickles - Boolean flag - if True will look for that last previous run position and tuning factor and start here
    method = 'thoth' - method name for construction of prior - recommend thoth construction
    
    Output
    ------
    Posterior
     - Initial thorium - values, 1𝜎, 65% credible level , 95% credible level
     - U-Th Ages - values, 1𝜎, 65% credible level , 95% credible level
     - Initial uranium  - values, 1𝜎, 65% credible level , 95% credible level
    """
    def __init__(self, Thor_KDE, Age_Maximum,
                 Age_Uncertainties, data, sample_name = 'SAMPLE_NAME',
                 n_chains = 3,
                 iterations = 50000,
                 burn_in = 10000,
                 Start_from_pickles = True,
                 method = 'thoth'):

        self.method = method
        self.data = data
        self.Thor_KDE = Thor_KDE
        self.burn_in = burn_in
        self.Age_Maximum = Age_Maximum
        self.Age_Uncertainties = Age_Uncertainties  # 1 sigma uncertainties (should be correct from model input)

        self.Th230_lam = 9.17055e-06   # Cheng et al. (2013)
        self.Th230_lam_err = 6.67e-09 # Cheng et al. (2013)
        self.U234_lam = 2.82203e-06 # Cheng et al. (2013)
        self.U234_lam_err = 1.494e-09 # Cheng et al. (2013)

        self.N_meas = data.shape[0]
        self.n_chains = n_chains
        self.Depths = data['Depths'].values
        self.Depths_err = data['Depths_err'].values
        self.iterations = iterations
        self.sample_name = sample_name
        self.Chain_Results = None
        self.Start_from_pickles = Start_from_pickles

        self.depths = self.data['Depths'].values

        # Short Hand measured activity ratios
        self.r08 = data['Th230_238U_ratios'].values
        self.r28 = data['Th232_238U_ratios'].values
        self.r48 = data['U234_U238_ratios'].values

        # Short Hand measured activity ratio uncertainties
        self.r08_err = data['Th230_238U_ratios_err'].values
        self.r28_err = data['Th232_238U_ratios_err'].values
        self.r48_err = data['U234_U238_ratios_err'].values

        # per-index tuning scales will use in the chain to get to the
        # desired acceptance rate
        self.tuning = {}
        for i in range(self.N_meas):
            self.tuning[f'Initial_Thorium_{i}']     = 1.0
            self.tuning[f'Th230_U238_ratios_{i}']   = float(self.r08_err[i])
            self.tuning[f'Th232_U238_ratios_{i}']   = float(self.r28_err[i])
            self.tuning[f'U234_U238_ratios_{i}']    = float(self.r48_err[i])

        self.keys = self.tuning.keys()
        self.rng  = np.random.default_rng()

        # --- precompute pairwise structures that
        # will be used in the stratigraphic likelihood calculation --- #
        N = self.N_meas
        I, J = np.triu_indices(N, k=1)
        self._IJ_I = I
        self._IJ_J = J

        errs = np.asarray(self.Age_Uncertainties, float)
        self._pair_sigma = np.hypot(errs[I], errs[J])

        d = self.depths
        self._pair_same = (d[J] == d[I])
        self._pair_diff = ~self._pair_same

        self.store_thin  = 5      # thin chain - easier to save
        self.store_dtype = np.float32

        # --- define Thor helpers FIRST ---
        def _thor_pdf(x):
            x = np.asarray(x, float)
            if hasattr(self.Thor_KDE, "pdf"): # If lognorm is used
                return self.Thor_KDE.pdf(x)
            elif callable(self.Thor_KDE):           # gaussian_kde
                return np.asarray(self.Thor_KDE(x), float)
            else:
                raise TypeError("Thor_KDE must be gaussian_kde or a scipy frozen rv.")
        self._thor_pdf = _thor_pdf

        def _thor_logpdf(x):
            px = np.clip(self._thor_pdf(x), 1e-300, None)
            return np.log(px)
        self._thor_logpdf = _thor_logpdf

        def _thor_rvs(n):
            if hasattr(self.Thor_KDE, "rvs"):       # lognorm etc.
                return np.asarray(self.Thor_KDE.rvs(size=n), float)
            # KDE: lazy-build inverse CDF if needed
            if getattr(self, "_thor_inv_cdf", None) is None:
                self._build_thor_inv_cdf()
            u = self.rng.random(n)
            return np.asarray(self._thor_inv_cdf(u), float)
        self._thor_rvs = _thor_rvs

        # optionally prebuild the inverse Cumulative denisty function from
        # Thoth - Helps with speed
        if self.method == 'thoth':
            self._build_thor_inv_cdf()

        # --- Set of move functions that the MCMC can call
        self._move_funcs = [
            ("Initial_Thorium",       self.Initial_Thorium_Move),
            ("Th230_U238_ratios",     self.Th230_U238_Move),
            ("Th232_U238_ratios",     self.Th232_U238_Move),
            ("U234_U238_ratios",      self.U234_U238_Move),
            ("PerSampleBlock",        self.PerSampleBlock_Move),   # NEW
            ("Smart_Order_Directed",  self.Smart_Order_Directed_Move),
        ]

        # --- variables needed for the smart move use
        self._depth_order = np.argsort(self.depths)  # shallow -> deep
        i_adj = self._depth_order[:-1]
        j_adj = self._depth_order[1:]
        self._adj_I = i_adj
        self._adj_J = j_adj
        self._adj_sigma = np.hypot(errs[i_adj], errs[j_adj])  # √(σ_i^2 + σ_j^2)

        # Cache for current ages - MCMC can get ages fast
        self._ages_cache = None

    # ---------------- utility for strat violations ----------------
    def _worst_adj_violation(self, ages):
        """
        Among adjacent (by depth) pairs (i -> shallow, j -> deeper), compute Δ = age[j] - age[i].
        Strat expects Δ >= 0. Return the (i, j, Δ, z) for the *most negative* z = Δ / σ_ij.
        If no violation (all z >= 0), return (None, None, None, None).
        """
        i = self._adj_I
        j = self._adj_J
        Δ = ages[j] - ages[i]
        z = Δ / np.maximum(self._adj_sigma, 1e-12)
        k = np.argmin(z)
        if z[k] < 0.0:
            return int(i[k]), int(j[k]), float(Δ[k]), float(z[k])
        return None, None, None, None

    def _desired_th_sign(self, i, j, idx):
        """
        For the worst adj violation pair (i->shallow, j->deeper, Δ<0):
        - If we move j (deeper too young), we want age↑ -> Th0↓ -> sign_th = -1
        - If we move i (shallow too old), we want age↓ -> Th0↑ -> sign_th = +1
        """
        return -1 if idx == j else +1

    # --------------------- age solvers --------------------------
    def U_series_age_equation(self, age: float, Th_initial: float,
                              Th232_ratio: float, U234_ratio: float,
                              Th230_ratio: float) -> float:
        """
        U‐series equation: returns f(age) = 0 when isotopic ratios match.
        """
        dlam = self.Th230_lam - self.U234_lam
        L1 = Th230_ratio
        L2 = Th232_ratio * Th_initial * np.exp(-self.Th230_lam * age)
        R1 = 1.0 - np.exp(-self.Th230_lam * age)
        if abs(dlam) < 1e-12:
            return np.nan
        R2 = (U234_ratio - 1.0) * (self.Th230_lam / dlam) * (1.0 - np.exp(-dlam * age))
        return L1 - L2 - R1 - R2

    # ---- Newton + brentq hybrid single age solver ----
    def _age_fun_and_deriv(self, a, Th_initial, Th232, U234, Th230):
        l230 = self.Th230_lam
        l234 = self.U234_lam
        dlam = l230 - l234
        e230 = np.exp(-l230 * a)
        ed   = np.exp(-dlam * a)
        f  = (Th230 - Th232 * Th_initial * e230 - (1.0 - e230)
              - (U234 - 1.0) * (l230 / dlam) * (1.0 - ed))
        fp = l230 * e230 * (Th232 * Th_initial - 1.0) - l230 * (U234 - 1.0) * ed
        return f, fp

    def _solve_age_single(self, Th_initial, Th232, U234, Th230, a0, amax, newton_max=8):
        amin = 1e-6  # small positive floor in your time units
        a = np.clip(a0 if np.isfinite(a0) else 0.5*amax, amin, amax)
        for _ in range(newton_max):
            f, fp = self._age_fun_and_deriv(a, Th_initial, Th232, U234, Th230)
            if not np.isfinite(f) or not np.isfinite(fp) or abs(fp) < 1e-14:
                break
            step = f / fp
            a_new = a - step
            if (a_new < 0.0) or (a_new > amax):
                a_new = np.clip(a - 0.5*step, 0.0, amax)
            if abs(a_new - a) < 1e-10 * max(1.0, a):
                return float(a_new)
            a = a_new

        def g(t):
            return self.U_series_age_equation(t, Th_initial, Th232, U234, Th230)

        left  = max(amin, a - 0.2*amax)
        right = min(amax, a + 0.2*amax)
        fL, fR = g(left), g(right)
        if np.isfinite(fL) and np.isfinite(fR) and fL*fR <= 0:
            return float(brentq(g, left, right, maxiter=100))

        grid = np.linspace(amin, amax, 16)
        vals = [g(t) for t in grid]
        for k in range(len(grid)-1):
            if np.isfinite(vals[k]) and np.isfinite(vals[k+1]) and vals[k]*vals[k+1] <= 0:
                return float(brentq(g, grid[k], grid[k+1], maxiter=100))
        return np.nan

    def ages_vector(self, Th_initial, U234, Th230, Th232, age_guess=None):
        N = len(Th_initial)
        ages = np.empty(N, dtype=float)
        if age_guess is None or np.isscalar(age_guess):
            guess = np.full(N, 0.5 * self.Age_Maximum, float)
        else:
            guess = np.asarray(age_guess, float)
            if guess.shape[0] != N:
                raise ValueError("age_guess must be length N")
        amax = float(self.Age_Maximum)
        for i in range(N):
            ages[i] = self._solve_age_single(Th_initial[i], Th232[i], U234[i], Th230[i],
                                             a0=guess[i], amax=amax)
        return ages

    # -------------------- priors  ---------------------------
    def Initial_Thorium_Prior(self, Initial_Thorium):
        dens = self._thor_pdf(Initial_Thorium)
        dens = np.clip(dens, 1e-300, None)
        return float(np.sum(np.log(dens)))

    def Initial_Thorium_MarginalPDF(self, x):
        return np.clip(self._thor_pdf(x), 1e-300, None)

    def _build_thor_inv_cdf(self, x_min=0, x_max=None, grid_points=1000):
        if hasattr(self.Thor_KDE, "rvs"):
            return
        if x_max is None:
            x_max = 50.0
        x_grid = np.linspace(x_min, x_max, grid_points)
        pdf_vals = np.clip(self._thor_pdf(x_grid), 1e-300, None)
        dx = x_grid[1] - x_grid[0]
        cdf_vals = np.cumsum(pdf_vals) * dx
        cdf_vals /= cdf_vals[-1]
        self._thor_inv_cdf = interp1d(cdf_vals, x_grid, bounds_error=False,
                                      fill_value=(x_min, x_max))

    def sample_one_initial_th(self):
        u = self.rng.random()
        return float(self._thor_inv_cdf(u))

    def ln_prior_initial_th(self, th_vec: np.ndarray) -> float:
        return float(np.sum(self._thor_logpdf(th_vec)))

    def ln_prior_ratios(self, U234, Th230, Th232) -> float:
        if np.any(U234 <= 0) or np.any(Th230 <= 0) or np.any(Th232 <= 0):
            return -np.inf
        # clip scales - should never have an age = 0.0
        s48 = np.clip(self.r48_err, 1e-12, np.inf)
        s08 = np.clip(self.r08_err, 1e-12, np.inf)
        s28 = np.clip(self.r28_err, 1e-12, np.inf)
        lp  = np.sum(norm.logpdf(U234, loc=self.r48, scale=s48))
        lp += np.sum(norm.logpdf(Th230, loc=self.r08, scale=s08))
        lp += np.sum(norm.logpdf(Th232, loc=self.r28, scale=s28))
        return lp

    def _age_violation(self, ages, min_age=1e-6):
        below = np.maximum(0.0, min_age - ages)  # penalize ages < min_age
        above = np.maximum(0.0, ages - self.Age_Maximum)
        return below + above

    def ln_prior(self, theta):
        th, U234, Th230, Th232 = theta
        ages = self.ages_vector(th, U234, Th230, Th232)
        if np.any(th <= 0) or np.any(U234 <= 0) or np.any(Th230 <= 0) or np.any(Th232 <= 0) or np.any(ages <= 0):
            return -np.inf
        lp_th  = float(np.sum(self._thor_logpdf(th)))
        lp_rat = self.ln_prior_ratios(U234, Th230, Th232)
        if not np.isfinite(lp_th + lp_rat):
            return -1e12
        if np.any(~np.isfinite(ages)):
            return (lp_th + lp_rat) - 1e12
        viol = self._age_violation(ages) / max(1.0, float(self.Age_Maximum))
        W = 5e4
        penalty = - W * np.sum(viol**2)
        return (lp_th + lp_rat) + penalty

    # ----------------- Make initial model vector guess ---------------------
    def Initial_Guesses_for_Model(self, max_attempts=10000):
        initial_thetas = []
        Th230_c = self.data['Th230_238U_ratios'].values
        Th232_c = self.data['Th232_238U_ratios'].values
        U234_c  = self.data['U234_U238_ratios'].values
        Th230_s = self.data['Th230_238U_ratios_err'].values
        Th232_s = self.data['Th232_238U_ratios_err'].values
        U234_s  = self.data['U234_U238_ratios_err'].values

        for chain in range(self.n_chains):
            theta_found = False
            for _ in range(max_attempts):
                Th_initial = self._thor_rvs(self.N_meas)
                Th230_in = np.random.normal(Th230_c, Th230_s)
                Th232_in = np.random.normal(Th232_c, Th232_s)
                U234_in  = np.random.normal(U234_c,  U234_s)
                theta = (Th_initial, U234_in, Th230_in, Th232_in)
                logp, ages = self.log_posterior(theta)
                if (np.any(~np.isfinite(ages)) or np.any(ages < 0) or np.any(ages > self.Age_Maximum)):
                    continue
                if np.isfinite(logp):
                    initial_thetas.append(theta)
                    theta_found = True
                    break
            if not theta_found:
                Th_initial_fb = np.ones(self.N_meas, dtype=float) * 0.8
                theta_fb = (Th_initial_fb, U234_c.copy(), Th230_c.copy(), Th232_c.copy())
                initial_thetas.append(theta_fb)
        return initial_thetas

    # ==============================================================
    # ==============================================================
    # ==============================================================
    # ==============================================================
    # ---------------- likelihoods and posterior -------------------
    # ==============================================================
    # ==============================================================
    # ==============================================================
    # ==============================================================
    def strat_likelihood(self, ages):
        I = self._IJ_I; J = self._IJ_J
        Δ = ages[J] - ages[I]
        σ = np.maximum(self._pair_sigma, 1e-12)
        same = self._pair_same
        diff = self._pair_diff
        ll_same = norm.logpdf(Δ[same], loc=0.0, scale=σ[same]).sum()
        z = Δ[diff] / σ[diff]
        ll_strat = log_ndtr(z).sum()  # robust log Φ(z)
        return ll_same + ll_strat

    def log_posterior(self, theta, ages_prev=None):
        th, U234, Th230, Th232 = theta
        ages = self.ages_vector(th, U234, Th230, Th232, age_guess=ages_prev)

        # hard rejections
        if (np.any(~np.isfinite(ages)) or np.any(ages <= 0) or
            np.any(th <= 0) or np.any(U234 <= 0) or np.any(Th230 <= 0) or np.any(Th232 <= 0)):
            return -np.inf, ages

        lp_th  = float(np.sum(self._thor_logpdf(th)))
        lp_rat = self.ln_prior_ratios(U234, Th230, Th232)
        if not np.isfinite(lp_th + lp_rat):
            return -1e12, ages

        viol = self._age_violation(ages) / max(1.0, float(self.Age_Maximum))
        penalty = -5e4 * np.sum(viol**2)
        ll = self.strat_likelihood(ages)
        return (lp_th + lp_rat + penalty + ll), ages


    def log_posterior_given_ages(self, theta, ages):
        th, U234, Th230, Th232 = theta
        if (np.any(th <= 0) or np.any(U234 <= 0) or np.any(Th230 <= 0) or np.any(Th232 <= 0) or
            np.any(~np.isfinite(ages)) or np.any(ages <= 0)):
            return -np.inf

        lp = float(np.sum(self._thor_logpdf(th))) + self.ln_prior_ratios(U234, Th230, Th232)
        if not np.isfinite(lp):
            return -1e12

        viol = self._age_violation(ages) / max(1.0, float(self.Age_Maximum))
        penalty = -5e4 * np.sum(viol**2)
        ll = self.strat_likelihood(ages)
        return lp + penalty + ll
    # ==============================================================
    # ==============================================================
    # ==============================================================
    # ==============================================================
    # ---------------- likelihoods and posterior  nd --------------
    # ==============================================================
    # ==============================================================
    # ==============================================================
    # ==============================================================

    ##############################################
    # ============= Moves for MCMC =============
    ##############################################

    def Th232_U238_Move(self, theta, tuning, index):
        _, _, _, Th232 = theta
        Th232_prime = Th232.copy()
        Th232_prime[index] += np.random.normal(0.0, tuning)
        return Th232_prime, True

    def Th230_U238_Move(self, theta, tuning, index):
        _, _, Th230, _ = theta
        Th230_prime = Th230.copy()
        Th230_prime[index] += np.random.normal(0.0, tuning)
        return Th230_prime, True

    def U234_U238_Move(self, theta, tuning, index):
        _, U234, _, _ = theta
        U234_prime = U234.copy()
        U234_prime[index] += np.random.normal(0.0, tuning)
        return U234_prime, True

    def Initial_Thorium_Move(self, theta, tuning, index):
        init_th, _, _, _ = theta
        init_th_prime = init_th.copy()
        if self.rng.random() < 0.05:  # global refresh from prior
            init_th_prime[index] = max(1e-12, float(self._thor_rvs(1)[0]))
        else:  # local log-normal step in log-space
            mu = np.log(max(init_th[index], 1e-12))
            init_th_prime[index] = np.exp(mu + self.rng.normal(0.0, tuning))
        return init_th_prime, True

    def PerSampleBlock_Move(self, theta, _tuning_unused, index):
        """
        Jointly perturb (InitTh, U234, Th230, Th232) at a single index.
        Uses existing per-parameter tuning scales for that index.
        """
        th_cur, U234_cur, Th230_cur, Th232_cur = theta

        def _scale(key, default):
            return float(self.tuning.get(key, default))

        k_th   = f'Initial_Thorium_{index}'
        k_u234 = f'U234_U238_ratios_{index}'
        k_t30  = f'Th230_U238_ratios_{index}'
        k_t32  = f'Th232_U238_ratios_{index}'

        s_th   = _scale(k_th,   0.05 * max(1e-6, th_cur[index]))
        s_u234 = _scale(k_u234, 0.002)
        s_t30  = _scale(k_t30,  0.002)
        s_t32  = _scale(k_t32,  0.002)

        d_th   = self.rng.normal(0.0, s_th)
        d_u234 = self.rng.normal(0.0, s_u234)
        d_t30  = self.rng.normal(0.0, s_t30)
        d_t32  = self.rng.normal(0.0, s_t32)

        th_new    = th_cur.copy();    th_new[index]    = max(1e-12, th_cur[index]    + d_th)
        U234_new  = U234_cur.copy();  U234_new[index]  = max(1e-6,  U234_cur[index]  + d_u234)
        Th230_new = Th230_cur.copy(); Th230_new[index] = max(1e-12, Th230_cur[index] + d_t30)
        Th232_new = Th232_cur.copy(); Th232_new[index] = max(1e-12, Th232_cur[index] + d_t32)

        return (th_new, U234_new, Th230_new, Th232_new), None

    def Smart_Order_Directed_Move(self, theta, _tuning_ignored=None, _index_ignored=None, drift_scale=1.0):
        """
        Informed move choice - select the out of order sample here
        """
        init_th, U234, Th230, Th232 = theta
        ages_here = self._ages_cache
        if ages_here is None:
            ages_here = self.ages_vector(init_th, U234, Th230, Th232)

        i, j, Δ, z = self._worst_adj_violation(ages_here)
        if i is None:
            idx = int(self.rng.integers(0, self.N_meas))
            s = 0
        else:
            si = float(self.Age_Uncertainties[i]); sj = float(self.Age_Uncertainties[j])
            idx = j if (sj > si) else i
            s = self._desired_th_sign(i, j, idx)

        th_new = init_th.copy()
        sigma  = float(self.tuning.get(f'Initial_Thorium_{idx}', 1.0))
        y_old  = float(np.log(max(th_new[idx], 1e-12)))
        mu_prop = y_old + (s * drift_scale * sigma)
        y_new  = float(self.rng.normal(mu_prop, sigma))
        th_new[idx] = float(np.exp(y_new))
        return th_new, idx, y_old, y_new, sigma, s, float(drift_scale)

    # ---------------- Calcuate U234_0 - initial ratio --------------
    def Initial_234U(self, theta):
        # Calculation of initial 234U/238U
        # Ratio is stored in output
        Initial_Th_mean, U234_ratios, Th230_ratios, Th232_ratios = theta
        Uages = self.ages_vector(Initial_Th_mean, U234_ratios, Th230_ratios, Th232_ratios)
        U234_initial = 1 + ((U234_ratios - 1) * np.exp(self.U234_lam * Uages))
        return U234_initial
        
    
    # Tuning function
    # Could set target_rate as global model variable - but it works here so just not going to bother at the moment.
    def adapt_tuning(self):
        # Gets the last chunk of chains for all variables
        # assess the acceptance rate of the chunk
        # if too high or too low then the tuning parameter is adapted
        
        target_rate = 0.234  # Target acceptance rate of 23.4% - not an exact thing but its a really nice rule-of-thumb to use - keeps the chains trundling along nicely
        for param in self.tuning:
            p = self.proposal_counts.get(param, 0)
            if p > 0:
                a = self.accept_counts.get(param, 0)
                rate = a / p
                if rate < target_rate:
                    self.tuning[param] *= 0.9
                else:
                    self.tuning[param] *= 1.1

    # Save pickles to file here
    # Can then grab them for running again if required
    def Save_Parameters_and_Tuning(self, theta, chain_id):
        tf_file    = f'tuning_{self.sample_name}_{chain_id}.pkl'
        theta_file = f'{self.sample_name}_theta_{chain_id}.pkl'
        with open(theta_file, 'wb') as f:
            pickle.dump(theta, f)
        with open(tf_file, 'wb') as f:
            pickle.dump(self.tuning, f)
    
    # Short hand function to update theta - model parameter vector
    def update_params(self, theta, move_name, new_value):
        init_th, U234, Th230, Th232 = theta
        if move_name == 'Initial_Thorium':
            init_th = new_value
        elif move_name == 'U234_U238_ratios':
            U234 = new_value
        elif move_name == 'Th232_U238_ratios':
            Th232 = new_value
        elif move_name == 'Th230_U238_ratios':
            Th230 = new_value
        return init_th, U234, Th230, Th232

    # --------------------- main MCMC ----------------------------
    def MCMC(self, theta, iterations, chain_id):
        """
        Full Function for the Markov Chain Monte Carlo (MCMC)
        (speed-optimized: cache logp and ages; use precomputed pairs)
        """
        start_time = time.time()
        Ndata = self.N_meas
        total_iterations = iterations + self.burn_in

        # load per-chain tuning if present
        tf_file = f'tuning_{self.sample_name}_{chain_id}.pkl'
        if os.path.exists(tf_file) and self.Start_from_pickles:
            with open(tf_file, 'rb') as f:
                self.tuning = pickle.load(f)

        # init counters
        self.proposal_counts = {k: 0 for k in self.keys}
        self.accept_counts   = {k: 0 for k in self.keys}

        # ----- initial evaluation - compute and cache ages here -----
        init_ages_guess = np.full(Ndata, 0.5 * self.Age_Maximum, dtype=float)
        logp_cur, ages_cur = self.log_posterior(theta, ages_prev=init_ages_guess)
        self._ages_cache = ages_cur

        # allocate storage for posterior samples (after burn-in)
        keep = iterations // self.store_thin + int(iterations % self.store_thin != 0)
        Ages_store            = np.zeros((keep, Ndata), dtype=self.store_dtype)
        Initial_Th_mean_store = np.zeros((keep, Ndata), dtype=self.store_dtype)
        U234_initial_store    = np.zeros((keep, Ndata), dtype=self.store_dtype)
        U234_ratios_store     = np.zeros((keep, Ndata), dtype=self.store_dtype)
        Th232_ratios_store    = np.zeros((keep, Ndata), dtype=self.store_dtype)
        Th230_ratios_store    = np.zeros((keep, Ndata), dtype=self.store_dtype)
        posterior_store       = np.zeros(keep,            dtype=self.store_dtype)

        sample_index = 0
        
        # Set up progress bar here
        pbar = tqdm(
            range(1, total_iterations + 1),
            desc=f"Chain {chain_id}",
            dynamic_ncols = True,
            leave =False,
            ncols = 100,
            disable = (chain_id != 0) or sys.stdout.isatty())
        
        FLUSH_EVERY = max(self.store_thin, 1000)
        for i in pbar:
            # 1) choose move & construct proposal
            move_name, move_func = random.choice(self._move_funcs)

            if move_name == "Smart_Order_Directed":
                (new_init_th, idx_used, y_old, y_new, sig, s_fwd, drift_scale) = move_func(theta, None, None)
                counter_key = f'Initial_Thorium_{idx_used}'
                self.proposal_counts[counter_key] = self.proposal_counts.get(counter_key, 0) + 1

                th_cur, U234_cur, Th230_cur, Th232_cur = theta
                theta_prop = (new_init_th, U234_cur, Th230_cur, Th232_cur)
                idx_moved  = idx_used

            else:
                # pick a coordinate to perturb for the chosen move
                idx = np.random.randint(0, Ndata)
                counter_key = f'{move_name}_{idx}'
                self.proposal_counts[counter_key] = self.proposal_counts.get(counter_key, 0) + 1

                # pass a scalar tuning; PerSampleBlock ignores it and uses per-parameter scales internally
                new_piece, _ = move_func(theta, self.tuning.get(counter_key, 0.0), idx)

                th_cur, U234_cur, Th230_cur, Th232_cur = theta
                if move_name == 'Initial_Thorium':
                    theta_prop = (new_piece, U234_cur, Th230_cur, Th232_cur)
                    idx_moved = idx
                elif move_name == 'U234_U238_ratios':
                    theta_prop = (th_cur, new_piece, Th230_cur, Th232_cur)
                    idx_moved = idx
                elif move_name == 'Th232_U238_ratios':
                    theta_prop = (th_cur, U234_cur, Th230_cur, new_piece)
                    idx_moved = idx
                elif move_name == 'Th230_U238_ratios':
                    theta_prop = (th_cur, U234_cur, new_piece, Th232_cur)
                    idx_moved = idx
                elif move_name == 'PerSampleBlock':
                    # new_piece is a 4-tuple of full arrays
                    theta_prop = new_piece
                    idx_moved = idx  # still only one age solve
                else:
                    theta_prop = theta
                    idx_moved = None

            # 2) cheap age update
            # Save time
            # Only shifting one parameter at a time for a single horizon so just need to recalc that age- everything else is the same
            
            if idx_moved is not None:
                ages_prop = ages_cur.copy()
                th_p, U234_p, Th230_p, Th232_p = theta_prop

                # ---- early invalid-ratio guard ----
                if (th_p[idx_moved] <= 0) or (U234_p[idx_moved] <= 0) or \
                   (Th230_p[idx_moved] <= 0) or (Th232_p[idx_moved] <= 0):
                    logp_prop = -np.inf
                else:
                    ages_prop[idx_moved] = self._solve_age_single(
                        th_p[idx_moved], Th232_p[idx_moved], U234_p[idx_moved], Th230_p[idx_moved],
                        a0=ages_cur[idx_moved], amax=self.Age_Maximum
                    )
                    if not np.isfinite(ages_prop[idx_moved]):
                        logp_prop = -np.inf
                    else:
                        logp_prop = self.log_posterior_given_ages(theta_prop, ages_prop)
            else:
                # Fail fallback; evaluate full posterior
                logp_prop, ages_prop = self.log_posterior(theta_prop, ages_prev=ages_cur)
            
            # Smart order moves are not symmetric so need to adapt the math here
            # so that we are true to the ergodic forward and backward symmetry of the
            # Markov Chain
            log_q_fwd = 0.0
            log_q_rev = 0.0
            if (move_name == "Smart_Order_Directed") and np.isfinite(logp_prop):
                inv = 1.0 / max(sig, 1e-12)
                log_q_fwd = -0.5 * ((y_new - (y_old + s_fwd * drift_scale * sig)) * inv)**2 \
                            - np.log(sig) - 0.5 * np.log(2 * np.pi)
                i2, j2, _, _ = self._worst_adj_violation(ages_prop)
                s_rev = 0 if (i2 is None) else self._desired_th_sign(i2, j2, idx_used)
                log_q_rev = -0.5 * ((y_old - (y_new + s_rev * drift_scale * sig)) * inv)**2 \
                            - np.log(sig) - 0.5 * np.log(2 * np.pi)

            # 3) MH accept/reject
            # Metropolis-Hasting acceptance proposal here
            accept = np.isfinite(logp_prop) and \
                     (np.log(self.rng.random()) < (logp_prop - logp_cur + (log_q_rev - log_q_fwd)))

            if accept:
                theta    = theta_prop
                logp_cur = logp_prop
                ages_cur = ages_prop
                self.accept_counts[counter_key] = self.accept_counts.get(counter_key, 0) + 1

            # keep cache in sync with the CURRENT state
            self._ages_cache = ages_cur

            # store after burn-in (with thinning)
            if i > self.burn_in and ((i - self.burn_in) % self.store_thin == 0):
                if sample_index < keep:
                    Ages_store[sample_index, :]            = ages_cur
                    Initial_Th_mean_store[sample_index, :] = theta[0]
                    U234_initial_store[sample_index, :]    = 1 + ((theta[1] - 1.0) * np.exp(self.U234_lam * ages_cur))
                    U234_ratios_store[sample_index, :]     = theta[1]
                    Th232_ratios_store[sample_index, :]    = theta[3]
                    Th230_ratios_store[sample_index, :]    = theta[2]
                    posterior_store[sample_index]          = logp_cur
                    sample_index += 1

            # periodic save + adapt
            if i > 50 and i % 1000 == 0:
                self.Save_Parameters_and_Tuning(theta, chain_id)
            
            # So bar updates
            if i % total_iterations == 0:
                phase = "burn-in" if i < self.burn_in else "sampling"
                props = sum(self.proposal_counts.values()) or 1
                accs  = sum(self.accept_counts.values())
                acc_rate = accs / props
                pbar.set_postfix_str(f"{phase}, acc={acc_rate:.3f}")
            
            if (i % FLUSH_EVERY == 0) or (i == total_iterations):
                pbar.update(i - pbar.n)

            if i > 50 and i % 1000 == 0 and i < self.burn_in:
                self.adapt_tuning()

        return (Ages_store, Initial_Th_mean_store, U234_ratios_store,
                Th232_ratios_store, Th230_ratios_store,
                posterior_store, U234_initial_store)

    # ---------------- HELPER FOR INITIALIZATION ------------------------
    def check_starting_parameters(self):
        """
        Return a list of length self.n_chains of starting θ tuples.
        If Start_from_pickles is True and *all* pickle files exist, load them; otherwise generate new ones.
        """
        if self.Start_from_pickles:
            loaded = []
            for chain_id in range(self.n_chains):
                fname = f'{self.sample_name}_theta_{chain_id}.pkl'
                if os.path.exists(fname):
                    with open(fname, "rb") as f:
                        loaded.append(pickle.load(f))
                else:
                    break
            if len(loaded) == self.n_chains:
                print("Loaded starting θ from pickles")
                return loaded
        print("Generating new starting θ’s")
        return self.Initial_Guesses_for_Model()

    # Run MCMC function.
    # All chains run in parrallel (n_jobs = -1)
    def Run_MCMC(self):
        iterations = self.iterations
        chain_ids = range(self.n_chains)
        all_thetas = self.check_starting_parameters()

        def run_chain(theta, chain_id):
            return self.MCMC(theta, self.iterations, chain_id)

        results = Parallel(n_jobs=-1)(
            delayed(run_chain)(theta, chain_id)
            for theta, chain_id in zip(all_thetas[:self.n_chains], chain_ids)
        )
        self.Chain_Results = results
        return self.Chain_Results

    # Helpher Call function to make sure we have chains to analyse
    def Ensure_Chain_Results(self):
        if self.Chain_Results is None:
            self.Chain_Results = self.Run_MCMC()
        return self.Chain_Results

    # Store the result in a dictionary so we can call stuff easily and plot
    def Get_Results_Dictionary(self):
        N_outputs = 7
        Results_ = self.Chain_Results
        if Results_ is None:
            self.Chain_Results = self.Run_MCMC()
            Results_ = self.Chain_Results

        z_vars = [f"z{i+1}" for i in range(N_outputs)]
        for chain_id in range(1, self.n_chains +1):
            for z_var in z_vars:
                vars()[f"{z_var}_{chain_id}"] = Results_[chain_id - 1][z_vars.index(z_var)]

        results_dict = []
        for chain_id, result in enumerate(Results_, start=1):
            result_dict = {}
            for var_name, value in zip(z_vars, result):
                result_dict[f"{var_name}_{chain_id}"] = value
            results_dict.append(result_dict)
        return results_dict

    # Plot posterior function - always nice to do should see some bouncing around of the chains about a center and all chains should be the same
    def Get_Posterior_plot(self):
        result_dicts = self.Get_Results_Dictionary()
        log_p = []
        for i in range(1, self.n_chains + 1):
            chain_dict = result_dicts[i-1]
            log_p.append(chain_dict[f"z6_{i}"])
        fig, ax = plt.subplots(1,1, figsize=(5,5))
        for i in range(self.n_chains):
            ax.plot(log_p[i], label=f'Chain {i + 1}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Posterior')
        ax.legend(frameon=True, loc=4, fontsize=10, ncol=2)

    def Get_Posterior_Values(self):
        result_dicts = self.Get_Results_Dictionary()
        log_p = []
        for i in range(1, self.n_chains + 1):
            chain_dict = result_dicts[i-1]
            log_p.append(chain_dict[f"z6_{i}"])
        return log_p
        
    def _summarize_draws(self, draws, conf=0.95, center="median"):
        """
        Robust posterior summary from draws.
        Returns:
        center_vals : chosen center per dimension
        95% confidences
        68% confidences
        sigma_1sd   : Symmetric est. (wouldnt use this for the initial thorium)
        Is really heavily tailed so confidences are best but for consistency we report all.
        """
        import numpy as np
    
        X = np.asarray(draws, float)
    
        # Drop rows with non-finite values
        if X.ndim == 1:
            X = X[np.isfinite(X)]
        elif X.ndim == 2:
            keep = np.isfinite(X).all(axis=1)
            X = X[keep]
        else:
            raise ValueError("draws must be 1D or 2D array")
    
        if X.size == 0:
            raise ValueError("No finite draws to summarize")
    
        # Center choice
        # Prefer median for center choice
        # if guassian mean = median = mode
        # but median covers the skewed potential of the initial thorium
        if center.lower() == "median":
            c = np.percentile(X, 50.0, axis=0)
        elif center.lower() == "mean":
            c = np.mean(X, axis=0)
        else:
            raise ValueError("center must be 'median' or 'mean'")
    
        # Equal-tailed CI at 'conf'
        alpha = 100.0 * (1.0 - conf)
        lo = np.percentile(X, alpha/2.0, axis=0)
        hi = np.percentile(X, 100.0 - alpha/2.0, axis=0)
    
        # Ensure monotonic ordering
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    
        # If chosen center lies outside [lo, hi] (can happen for center='mean'), recenter to median
        mask = (c < lo) | (c > hi)
        if np.any(mask):
            c_med = np.percentile(X, 50.0, axis=0)
            c = np.where(mask, c_med, c)
    
        # Non-negative errors relative to the (possibly adjusted) center
        low_err95  = np.maximum(c - lo, 0.0)
        high_err95 = np.maximum(hi - c, 0.0)
 
        alpha68 = 100.0 * (1.0 - 0.68)
        lo68 = np.percentile(X, alpha68/2.0, axis=0)
        hi68 = np.percentile(X, 100.0 - alpha68/2.0, axis=0)
    
        # Ensure monotonic ordering (paranoia)
        lo68, hi68 = np.minimum(lo68, hi68), np.maximum(lo68, hi68)
        
        low_err68  = np.maximum(c - lo68, 0.0)
        high_err68 = np.maximum(hi68 - c, 0.0)
    
        # Symmetric 1σ from the 16th/84th percentiles (always ≥ 0)
        p16 = np.percentile(X, 15.865525393145708, axis=0)
        p84 = np.percentile(X, 84.13447460685429, axis=0)
        sigma_1sd = 0.5 * (p84 - p16)
        sigma_1sd = np.maximum(sigma_1sd, 0.0)
    
        return c, (low_err95, high_err95), (low_err68, high_err68),  sigma_1sd


    def Get_Useries_Ages(self, return_sigma=False):
        result_dicts = self.Get_Results_Dictionary()
        chains = [result_dicts[i-1][f"z1_{i}"] for i in range(1, self.n_chains + 1)]
        all_draws = np.vstack(chains)  # (n_draws, N_meas)
        center, (low_err95, high_err95),(low_err68, high_err68), sigma = self._summarize_draws(all_draws, conf=0.95,    center="median")
        return (center, [low_err95, high_err95], [low_err68, high_err68], sigma) if return_sigma else (center, [low_err68, high_err68])
    
    def Get_Initial_Thoriums(self, return_sigma=False):
        result_dicts = self.Get_Results_Dictionary()
        chains = [result_dicts[i-1][f"z2_{i}"] for i in range(1, self.n_chains + 1)]
        all_draws = np.vstack(chains)
        center, (low_err95, high_err95),(low_err68, high_err68), sigma = self._summarize_draws(all_draws, conf=0.95,    center="median")
        return (center, [low_err95, high_err95], [low_err68, high_err68], sigma) if return_sigma else (center, [low_err68, high_err68])
    
    def Get_234U_initial(self, return_sigma=False):
        result_dicts = self.Get_Results_Dictionary()
        chains = [result_dicts[i-1][f"z7_{i}"] for i in range(1, self.n_chains + 1)]
        all_draws = np.vstack(chains)
        center, (low_err95, high_err95),(low_err68, high_err68), sigma = self._summarize_draws(all_draws, conf=0.95,    center="median")
        return (center, [low_err95, high_err95], [low_err68, high_err68], sigma) if return_sigma else (center, [low_err68, high_err68])
      # ========================================
     # ========================================
   # Individual save function for parameters
   # ========================================
      # ========================================
        # ========================================

    def Save_Initial_Thorium(self):
        Model_Ini_Thorium, Model_Ini_Thorium_err = self.Get_Initial_Thoriums()
        df_thor = pd.DataFrame({
            "Depth_Meas" : self.data['Depths'].values,
            "Depth_Meas_err" : self.data['Depths_err'].values,
            "Model_initial_th" : Model_Ini_Thorium,
            "M_Initial_Thorium_err1" : Model_Ini_Thorium_err[0],
            "M_Initial_Thorium_err2" : Model_Ini_Thorium_err[1]
        })
        df_thor.to_excel(f'{self.sample_name}_Initial_Thoriums.xlsx')

    def Save_234U_Initial(self):
        U0, U0_err = self.Get_234U_initial()
        df = pd.DataFrame({
            "Depth_Meas": self.data['Depths'].values,
            "Depth_Meas_err": self.data['Depths_err'].values,
            "Model_initial_234U": U0,
            "Model_initial_234U_err1": U0_err[0],
            "Model_initial_234U_err2": U0_err[1],
        })
        df.to_excel(f'{self.sample_name}_Initial_234U.xlsx')

    def Save_Useries_Ages(self):
        U_series_ages, U_series_ages_err = self.Get_Useries_Ages()
        df_ad = pd.DataFrame({
            "Depth_Meas" : self.data['Depths'].values,
            "Depth_Meas_err" : self.data['Depths_err'].values,
            "U_ages" : U_series_ages,
            "U_Age_low" : U_series_ages_err[0],
            "U_Age_high" : U_series_ages_err[1]
        })
        df_ad.to_excel(f'{self.sample_name}_U_Series_Ages.xlsx')
        
      # ========================================
     # ========================================
   # Individual save function for parameters
   # ========================================
      # ========================================
        # ========================================

    # ---------------- diagnostics  --------------------
    # Run if selected not always done
    # Rule of thumnb is that the Gelman-Rubin statistic for
    # the suite of chains that are run should be ~1.
    # This suggests good mixing and interchain relationships
    # E.g., the chains so similar parameter values and variances
    # this is an expectation of good mixing and arrival at the "True"
    # posterior.
    def Gelman_Rubin(self, chain_list):
        n_ch = len(chain_list)
        n = chain_list[0].shape[0]
        stacked_chains = np.stack(chain_list, axis=0)
        chain_means = np.mean(stacked_chains, axis=1)
        grand_mean = np.mean(chain_means, axis=0)
        B = (n / (n_ch - 1)) * np.sum((chain_means - grand_mean)**2, axis=0)
        W = np.mean(np.var(stacked_chains, axis=1, ddof=1), axis=0)
        var_plus = ((n - 1) / n) * W + (1 / n) * B
        R_hat = np.sqrt(var_plus / W)
        return R_hat

    # Run this for initial thorium chain stats
    def In_Thor_Chain_Stats(self):
        result_dicts = self.Get_Results_Dictionary()
        In_Thorium_chains = []
        for i in range(self.n_chains):
            chain_dict = result_dicts[i]
            In_Thorium_chains.append(np.array(chain_dict[f"z2_{i+1}"]))
        R_hat = self.Gelman_Rubin(In_Thorium_chains)
        return R_hat
        
    # Run this for U-Th age chain stats
    def Useries_Age_Chain_Stats(self):
        result_dicts = self.Get_Results_Dictionary()
        Uage_chains = []
        for i in range(self.n_chains):
            chain_dict = result_dicts[i]
            Uage_chains.append(np.array(chain_dict[f"z1_{i+1}"]))
        R_hat = self.Gelman_Rubin(Uage_chains)
        return R_hat
        
    # Get the full Summary dataframe
    def SummaryDataFrame(self):
        # pull symmetric 1σ directly from draws
        U_series_ages, U_series_err95, U_series_err68, U_series_sigma = self.Get_Useries_Ages(return_sigma=True)
        Model_Ini_uranium, Model_Ini_uranium_err95, Model_Ini_uranium_err68, U0_sigma = self.Get_234U_initial(return_sigma=True)
        Model_Ini_Thorium, Model_Ini_Thorium_err95, Model_Ini_Thorium_err68, Th0_sigma =   self.Get_Initial_Thoriums(return_sigma=True)
    
        df_all = pd.DataFrame({
            "Depth_Meas"              : self.data['Depths'].values,
            "Depth_Meas_err"          : self.data['Depths_err'].values,
            "age"                     : U_series_ages,
            "age_err"                 : U_series_sigma,      # symmetric 1σ (p16–p84) est.
            "initial thorium"         : Model_Ini_Thorium,
            "initial thorium err"     : Th0_sigma,           # symmetric 1σ estimtae
            "initial uranium"         : Model_Ini_uranium,
            "initial uranium err"     : U0_sigma             # symmetric 1σ est.
        })
    
        # ADD 68% and 95% confidence levels here
        df_all["age_low68"]  = U_series_err68[0]
        df_all["age_high68"] = U_series_err68[1]
        df_all["age_low95"]  = U_series_err95[0]
        df_all["age_high95"] = U_series_err95[1]
        df_all["Th0_low68"]  = Model_Ini_Thorium_err68[0]
        df_all["Th0_high68"] = Model_Ini_Thorium_err68[1]
        df_all["Th0_low95"]  = Model_Ini_Thorium_err95[0]
        df_all["Th0_high95"] = Model_Ini_Thorium_err95[1]
        df_all["U0_low68"]   = Model_Ini_uranium_err68[0]
        df_all["U0_high68"]  = Model_Ini_uranium_err68[1]
        df_all["U0_low95"]   = Model_Ini_uranium_err95[0]
        df_all["U0_high95"]  = Model_Ini_uranium_err95[1]
    
        df_all.to_csv(f'{self.sample_name}_ibis_summary.csv')
        
