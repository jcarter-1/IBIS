import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde, norm, uniform, lognorm
from scipy.special import expit
from scipy.interpolate import interp1d
import matplotlib as mpl
from tqdm import tqdm, tnrange, tqdm_notebook
import dill as pickle
import time
import os
from joblib import Parallel, delayed
import random
import warnings
from scipy.optimize import brentq


class IBIS_MCMC:
    def __init__(self, Thor_KDE, Age_Maximum,
                 Age_Uncertainties, data, sample_name = 'SAMPLE_NAME',
                 n_chains = 3,
                 iterations = 50000,
                 burn_in = 10000,
                 Start_from_pickles = True):
        
        self.data = data
        self.Thor_KDE = Thor_KDE
        if self.Thor_KDE is None:
            # If there is no prior
            # We will assume a uniform prior
            # Th ~U(0, 200)
            self.Thor_KDE = uniform(0, 500)
        self.burn_in = burn_in
        self.Age_Maximum = Age_Maximum
        self.Age_Uncertainties = Age_Uncertainties # 1 sigma uncertainties
        self.Th230_lam = 9.1577e-06 # Cheng et al. (2013)
        self.Th230_lam_err = 1.3914e-08 # Cheng et al. (2013)
        self.U234_lam = 2.8263e-06 # Cheng et al. (2013)
        self.U234_lam_err = 2.8234e-09 # Cheng et al. (2013)
        self.N_meas = data.shape[0]
        self.n_chains = n_chains
        self.Depths = data['Depths'].values
        self.Depths_err = data['Depths_err'].values
        self.iterations = iterations
        self.sample_name = sample_name
        self.Chain_Results = None
        self.Start_from_pickles = Start_from_pickles
        self.depths = self.data['Depths'].values
        
        # Short Hand ratios
        self.r08 = data['Th230_238U_ratios'].values
        self.r28 = data['Th232_238U_ratios'].values
        self.r48 = data['U234_U238_ratios'].values
        # Short Hand uncertainties
        self.r08_err = data['Th230_238U_ratios_err'].values
        self.r28_err = data['Th232_238U_ratios_err'].values
        self.r48_err = data['U234_U238_ratios_err'].values
        
                                                  
        self.tuning = {}
        for i in range(self.N_meas):
            self.tuning[f'Initial_Thorium_{i}'] = 1
            self.tuning[f'Th230_U238_ratios_{i}'] = self.r08_err[i]
            self.tuning[f'Th232_U238_ratios_{i}'] = self.r28_err[i]
            self.tuning[f'U234_U238_ratios_{i}'] = self.r48_err[i]

                                            
        self.keys = self.tuning.keys()
        self.rng = np.random.default_rng()
        self._build_thor_inv_cdf()
        self._move_funcs = [("Initial_Thorium", self.Initial_Thorium_Move),
                            ("Th230_U238_ratios", self.Th230_U238_Move),
                            ("Th232_U238_ratios", self.Th232_U238_Move),
                            ("U234_U238_ratios", self.U234_U238_Move)]
                            
                
    def U_series_age_equation(self,
                              age: float,
                              Th_initial: float,
                              Th232_ratio: float,
                              U234_ratio: float,
                              Th230_ratio: float) -> float:
        """
        The U‐series equation: returns f(age) = 0 when the isotopic ratios match.
        f(age) = Th230/238U_meas
                 - Th232/238U_meas * Th_initial * exp(-Th230_lam * age)
                 - [1 - exp(-Th230_lam * age)]
                 - [ (U234/238U_meas - 1) * (Th230_lam/(Th230_lam - U234_lam)) * (1 - exp(-(Th230_lam - U234_lam)*age)) ]
        """
        dlam = self.Th230_lam - self.U234_lam
        L1 = Th230_ratio
        L2 = Th232_ratio * Th_initial * np.exp(-self.Th230_lam * age)
        R1 = 1.0 - np.exp(-self.Th230_lam * age)
        if abs(dlam) < 1e-12:
            # This should never happen for fixed decay constants, but guard anyway
            return np.nan
        R2 = (U234_ratio - 1.0) * (self.Th230_lam / dlam) * (1.0 - np.exp(-dlam * age))
        return L1 - L2 - R1 - R2

    def calculate_age(self,
                      Th_initial: float,
                      Th232_ratio: float,
                      U234_ratio: float,
                      Th230_ratio: float,
                      age_guess: float = None) -> float:
        """
        Solve U_series_age_equation(age)=0 for age, using fsolve.
        If fsolve fails or does not converge, return np.nan.
        """
        if age_guess is None:
            age_guess = 0.5 * self.Age_Maximum

        func = lambda a: self.U_series_age_equation(a,
                                                    Th_initial,
                                                    Th232_ratio,
                                                    U234_ratio,
                                                    Th230_ratio)
        # Suppress the “not making good progress” warning from fsolve
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="The iteration is not making good progress",
                                    category=RuntimeWarning)
            sol, info, ier, mesg = fsolve(func,
                                          age_guess,
                                          full_output=True,  # get ier code
                                          maxfev=200)        # allow up to 200 func evals

        if ier != 1 or not np.isfinite(sol[0]):
            return np.nan
        else:
            return float(sol[0])


    def calculate_age2(self,
                      Th_initial: float,
                      Th232_ratio: float,
                      U234_ratio: float,
                      Th230_ratio: float,
                      age_guess: float = None) -> float:
        """
        Solve U_series_age_equation(age)=0 for age in [0, Age_Maximum] using brentq.
        If no valid bracket is found, return np.nan.
        """
        f = lambda a: self.U_series_age_equation(a, Th_initial, Th232_ratio, U234_ratio, Th230_ratio)

        a_lo = 0.0
        a_hi = float(self.Age_Maximum)

        # coarse grid to find a sign change
        K = 128
        grid = np.linspace(a_lo, a_hi, K + 1)
        fvals = np.empty_like(grid)
        for k, x in enumerate(grid):
            fx = f(x)
            if not np.isfinite(fx):
                return np.nan
            fvals[k] = fx

        # find first adjacent pair with opposite signs
        for k in range(K):
            y0, y1 = fvals[k], fvals[k+1]
            if np.signbit(y0) != np.signbit(y1):
                try:
                    root = brentq(f, grid[k], grid[k+1], maxiter=100, xtol=1e-10, rtol=1e-12)
                except Exception:
                    return np.nan
                if not np.isfinite(root):
                    return np.nan
                # extra safety: clamp tiny negatives from numerics
                if root < 0:
                    if root > -1e-9:
                        root = 0.0
                    else:
                        return np.nan
                if root > self.Age_Maximum:
                    return np.nan
                return float(root)

        # no bracket found → treat as invalid
        return np.nan
        
    def ages_vector(self,
                    Th_initial: np.ndarray,
                    U234: np.ndarray,
                    Th230: np.ndarray,
                    Th232: np.ndarray) -> np.ndarray:
        """
        Vectorized wrapper over calculate_age. Returns array of length N_meas.
        """
        N = len(Th_initial)
        ages = np.empty(N)
        for i in range(N):
            ages[i] = self.calculate_age(Th_initial[i],
                                         Th232[i],
                                         U234[i],
                                         Th230[i],
                                         age_guess=0.5 * self.Age_Maximum)
        return ages
            

    def Initial_Thorium_Prior(self, Initial_Thorium):
        # 1) evaluate the KDE at all requested points
        #    gaussian_kde.__call__(x) returns an array of densities for each x
        dens = self.Thor_KDE(Initial_Thorium)
        
        integral = np.trapz(dens, )
        # 2) clip to avoid log(0)
        dens = np.clip(dens, 1e-300, None)
        
        # 3) sum the log‐densities
        lp_total = np.sum(np.log(dens))
        
        return lp_total
        
    def Initial_Thorium_MarginalPDF(self, x):
        """
        Return the marginal KDE density for init_th[i] at value x.
        If you built Thor_KDE as a multivariate KDE over all N coords,
        you must extract the i‐th marginal. For simplicity, assume you
        have N separate 1D KDEs, one per coordinate.
        """
        return np.clip(self.Thor_KDE(x), 1e-300, None)
            
    def _build_thor_inv_cdf(self,
                             x_min: float = 0,
                             x_max: float = None,
                             grid_points: int = 1000):
        """
        Precompute the inverse‐CDF (quantile function) for the 1D KDE on initial‐Th.
        We store self._thor_inv_cdf, so later we only need one uniform draw per index.
        """
        if x_max is None:
            x_max = 50

        x_grid = np.linspace(x_min, x_max, grid_points)
        pdf_vals = self.Thor_KDE(x_grid)
        # If the KDE ever returns zero, clip it to a tiny positive value so cdf doesn’t stall
        pdf_vals = np.clip(pdf_vals, 1e-300, None)

        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]
        self._thor_inv_cdf = interp1d(cdf_vals, x_grid,
                                      bounds_error=False,
                                      fill_value=(x_min, x_max))

    def sample_one_initial_th(self) -> float:
        """
        Draw a single initial‐Th value via inverse‐CDF sampling.
        """
        u = self.rng.random()
        return float(self._thor_inv_cdf(u))

    def ln_prior_initial_th(self, th_vec: np.ndarray) -> float:
        """
        Sum of log‐KDE‐densities for each element in th_vec, with floor to avoid log(0).
        """
        dens_vals = np.clip(self.Thor_KDE(th_vec), 1e-300, None)
        return np.sum(np.log(dens_vals))

    def ln_prior_ratios(self,
                        U234: np.ndarray,
                        Th230: np.ndarray,
                        Th232: np.ndarray) -> float:
        """
        Normal priors around measured means for U234, Th230, Th232 ratios.
        If any ratio ≤0, return -inf immediately.
        """
        if np.any(U234 <= 0) or np.any(Th230 <= 0) or np.any(Th232 <= 0):
            return -np.inf

        lp = 0.0
        # U234 prior
        lp += np.sum(norm.logpdf(U234,
                                 loc=self.data['U234_U238_ratios'].values,
                                 scale=self.data['U234_U238_ratios_err'].values))
        # Th230 prior
        lp += np.sum(norm.logpdf(Th230,
                                 loc=self.data['Th230_238U_ratios'].values,
                                 scale=self.data['Th230_238U_ratios_err'].values))
        # Th232 prior
        lp += np.sum(norm.logpdf(Th232,
                                 loc=self.data['Th232_238U_ratios'].values,
                                 scale=self.data['Th232_238U_ratios_err'].values))
                                 

        return lp

    def ln_prior(self, theta):
        th, U234, Th230, Th232 = theta
        ages = self.ages_vector(th, U234, Th230, Th232)

        # hard gates
        if np.any(np.isnan(ages)):
            return -np.inf
        if np.any(ages < 0) or np.any(ages > self.Age_Maximum):
            return -np.inf
        if np.any(th <= 0) or np.any(U234 <= 0) or np.any(Th230 <= 0) or np.any(Th232 <= 0):
            return -np.inf

        lp = 0.0
        # KDE prior on initial Th
        lp_th = self.ln_prior_initial_th(th)
        if not np.isfinite(lp_th):
            return -np.inf
        lp += lp_th

        # measured ratio priors (optional but recommended)
        lp_rat = self.ln_prior_ratios(U234, Th230, Th232)
        if not np.isfinite(lp_rat):
            return -np.inf
        lp += lp_rat

        return lp



    def Initial_Guesses_for_Model(self, max_attempts=1000):
        """
        Draw self.n_chains valid initial thetas by rejection sampling on the age‐constraints.
        If we hit max_attempts without finding a fully valid candidate, we accept the
        last one (so the chain can still start) but it will carry the huge improper‐age prior.
        """
        initial_thetas = []
    
        for chain in range(self.n_chains):
            logp = -np.inf

            while logp == -np.inf:
    
                # 1) propose everything
                Th_initial = np.array([self.sample_one_initial_th() for _ in range(self.N_meas)])

                Th230_in = np.random.normal(
                    self.data['Th230_238U_ratios'].values,
                    self.data['Th230_238U_ratios_err'].values
                )
                Th232_in = np.random.normal(
                    self.data['Th232_238U_ratios'].values,
                    self.data['Th232_238U_ratios_err'].values
                )
                U234_in = np.random.normal(
                    self.data['U234_U238_ratios'].values,
                    self.data['U234_U238_ratios_err'].values
                )
    
                theta = (Th_initial, U234_in, Th230_in, Th232_in)
                # 2) compute ages for this candidate
                ages = self.ages_vector(Th_initial, U234_in, Th230_in, Th232_in)
                ages = self.ages_vector(Th_initial, U234_in, Th230_in, Th232_in)
                if (np.any(~np.isfinite(ages)) or
                    np.any(ages < 0) or
                    np.any(ages > self.Age_Maximum)):
                    continue

                # 4) if ages are OK, compute the true prior
                logp  = self.ln_prior(theta)
                if logp != -np.inf:
                    initial_thetas.append(theta)

    
        return initial_thetas
            
    def strat_likelihood2(self, ages):
        """
        Stratigraphic log-likelihood that:
          • penalizes out-of-order ages at different depths
          • *constrains* same-depth ages to be equal (within uncertainty)
        """
        LL = 0.0
        n = len(ages)
        zeta = 10.0
        for i in range(n):
            for j in range(i+1, n):
                diff = ages[j] - ages[i]
                σ_ij = np.sqrt(self.Age_Uncertainties[i]**2 +
                               self.Age_Uncertainties[j]**2)

                if self.depths[j] == self.depths[i]:
                    # Same depth → ages should coincide
                    # Model diff ~ Normal(0, σ_ij)
                    LL += norm.logpdf(diff, loc=0.0, scale=σ_ij)

                else:
                    # Strict stratigraphy only when depths differ
                    if diff >= 0:
                        # no penalty for properly ordered
                        LL += 0.0
                    elif abs(diff) > σ_ij:
                        # large violation → full uncertainty
                        LL += norm.logcdf(diff / σ_ij)
                    else:
                        # small violation → softened by zeta
                        LL += norm.logcdf(diff / (zeta * σ_ij))

        return LL
        
    def strat_likelihood(self, ages):
        N = len(ages)
        errs = self.Age_Uncertainties

        # build all i<j arrays
        I, J = np.triu_indices(N, k=1)
        Δ   = ages[J] - ages[I]
        σ   = np.hypot(errs[I], errs[J])

        # same vs different depth masks
        same = (self.depths[J] == self.depths[I])
        diff = ~same

        # co‐depth equality terms
        ll_same = norm.logpdf(Δ[same], loc=0, scale=σ[same]).sum()
        # stratigraphy terms: one‐sided tail for all i≠j
        ll_strat = norm.logcdf(Δ[diff] / σ[diff]).sum()

        return ll_same + ll_strat


    def ln_likelihood(self, theta: tuple) -> float:
        Th_initial, U234, Th230, Th232 = theta
        ages = self.ages_vector(Th_initial, U234, Th230, Th232)
    
        if np.any(np.isnan(ages)):
            return -np.inf
        if np.any(ages < 0) or np.any(ages > self.Age_Maximum):
            return -np.inf
        else:
            return self.strat_likelihood(ages)

    def log_posterior(self, theta: tuple) -> float:
        """
        Compute joint log‐posterior = ln_prior + ln_likelihood.
        """
        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.ln_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

            
    ##############################################
    # ============= Moves for MCMC ============= #
    ##############################################
    def Th232_U238_Move(self, theta, tuning, index):
        init_th, U234, Th230, Th232 = theta
        logp_cur = self.log_posterior(theta)
        
        
        Th232_prime = Th232.copy()
        
        Th232_prime[index] += np.random.normal(0, tuning)
        
        theta_prime = init_th, U234, Th230, Th232_prime
        
        logp_prop = self.log_posterior(theta_prime)
        if not np.isfinite(logp_prop) or np.any(Th232_prime <= 0):
            return Th232, False
            
            
        u = np.random.rand()
        if np.log(u) < logp_prop - logp_cur:
            return Th232_prime, True
        else:
            return Th232, False
            

    def Th230_U238_Move(self, theta, tuning, index):
        init_th, U234, Th230, Th232 = theta
        logp_cur = self.log_posterior(theta)
        
        
        Th230_prime = Th230.copy()
        
        Th230_prime[index] += np.random.normal(0, tuning)
        
        theta_prime = init_th, U234, Th230_prime, Th232
        
        logp_prop = self.log_posterior(theta_prime)
        if not np.isfinite(logp_prop) or np.any(Th230_prime <= 0):
            return Th230, False
        
        u = np.random.rand()
        if np.log(u) < logp_prop - logp_cur:
            return Th230_prime, True
        else:
            return Th230, False
            
    def U234_U238_Move(self, theta, tuning, index):
        init_th, U234, Th230, Th232 = theta
        logp_cur = self.log_posterior(theta)
        
        
        U234_prime = U234.copy()
        
        U234_prime[index] += np.random.normal(0, tuning)
        
        theta_prime = init_th, U234_prime, Th230, Th232
        
        logp_prop = self.log_posterior(theta_prime)
        if not np.isfinite(logp_prop) or np.any(U234_prime <= 0):
            return U234, False
            
            
        u = np.random.rand()
        if np.log(u) < logp_prop - logp_cur:
            return U234_prime, True
        else:
            return U234, False
    
            
    def Initial_Thorium_Move(self, theta, tuning, index):
        """
        Propose a change to init_th[index], using a nonparametric KDE prior.
        - With 5% probability: global refresh (draw from marginal KDE).
        - Otherwise: local log-normal MH step on log(init_th[index]).

        We must forbid ±Inf or NaN at every stage.
        """
        init_th, U234, Th230, Th232 = theta

        logp_cur = self.log_posterior(theta)
        
        init_th_prime = init_th.copy()
        
        init_th_prime[index] += np.random.normal(0, tuning)
        
        theta_prime = init_th_prime, U234, Th230, Th232
        
        logp_prop = self.log_posterior(theta_prime)
        if not np.isfinite(logp_prop) or np.any(init_th_prime <= 0):
            return init_th, False
            
        u = np.random.rand()
        if np.log(u) < logp_prop - logp_cur:
            return init_th_prime, True
        else:
            return init_th, False
                

    def Initial_234U(self, theta):
        Initial_Th_mean, U234_ratios, \
        Th230_ratios, Th232_ratios = theta

        Uages = self.ages_vector(Initial_Th_mean, U234_ratios,
        Th230_ratios, Th232_ratios)
        

        U234_initial  = 1 + ((U234_ratios - 1) * np.exp(self.U234_lam * Uages))
        
        return U234_initial
        
    def adapt_tuning(self):
        target_rate = 0.234 # Target acceptance rate of 23.4%
        
        for param in self.tuning:
            p = self.proposal_counts.get(param, 0)
            if p > 0:
                a = self.accept_counts.get(param,0)
                rate  = a/p
                if rate < target_rate:
                    self.tuning[param] *= 0.9
                else:
                    self.tuning[param] *= 1.1
    
    def Save_Parameters_and_Tuning(self, theta, chain_id):
        tf_file = f'tuning_{self.sample_name}_{chain_id}.pkl'
        theta_file = f'{self.sample_name}_theta_{chain_id}.pkl'
        # 1) save the current theta
        with open(theta_file, 'wb') as f:
            pickle.dump(theta, f)

        # 2) save tuning_factors
        with open(tf_file, 'wb') as f:
            pickle.dump(self.tuning, f)
           
    def update_params(self, theta, move_name, new_value):
        """
        Helper Function for updating parameters
        """
        
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
        
        
    def MCMC(self, theta, iterations, chain_id):
        """
        Full Function for the Markov Chain Monte Carlo (MCMC)
        """
        start_time = time.time()
        # Unpack Theta
        Initial_Th_mean, U234_ratios, Th230_ratios, Th232_ratios = theta
    
        Ndata = self.N_meas
        total_iterations = iterations + self.burn_in  # total iterations including burn-in
    
        # 2) (Re‐)load per‐chain tuning & Σ if you want; otherwise skip
        tf_file = f'tuning_{self.sample_name}_{chain_id}.pkl'
        if os.path.exists(tf_file) and self.Start_from_pickles:
            with open(tf_file,'rb') as f:
                self.tuning_dict = pickle.load(f)

        # 3) init counters & histories
        self.proposal_counts = {k:0 for k in self.keys}
        self.accept_counts   = {k:0 for k in self.keys}
    
        # Initial model ages
        model_ages = self.ages_vector(Initial_Th_mean, U234_ratios, Th230_ratios, Th232_ratios)
    
        # Preallocate storage arrays for samples AFTER burn-in only
        Ages_store = np.zeros((iterations, Ndata))
        Initial_Th_mean_store = np.zeros((iterations, Ndata))
        U234_initial_store = np.zeros((iterations, Ndata))
        U234_ratios_store = np.zeros((iterations, Ndata))
        Th232_ratios_store = np.zeros((iterations, Ndata))
        Th230_ratios_store = np.zeros((iterations, Ndata))
        posterior_store = np.zeros(iterations)

        target_accept_rate = 0.234
    
        sample_index = 0  # Counter for samples after burn-in
    
        for i in range(1, total_iterations + 1):
        
            move_name, move_func = random.choice(self._move_funcs)
            
            index = np.random.randint(0, self.N_meas)
            specific_counter_name = f'{move_name}_{index}'
            self.proposal_counts[specific_counter_name] +=1
            
            new_value, accepted = move_func(theta, self.tuning[specific_counter_name],
            index)
            
            if accepted:
                self.accept_counts[specific_counter_name] +=1
                theta = self.update_params(theta, move_name, new_value)

            # Store samples only after burn-in
            if i > self.burn_in:
                # Save the current state to our storage arrays
               
                Ages_store[sample_index, :] = self.ages_vector(theta[0], theta[1], theta[2], theta[3])
                Initial_Th_mean_store[sample_index, :] = theta[0]
                U234_initial_store[sample_index, :] = self.Initial_234U(theta)
                U234_ratios_store[sample_index, :] = theta[1]
                Th232_ratios_store[sample_index, :] = theta[2]
                Th230_ratios_store[sample_index, :] = theta[3]
                posterior_store[sample_index] = self.log_posterior(theta)
                sample_index += 1
                
            """
            Chain Behaviour and Updates here
            """
            if i > 50 and i % 1000 == 0:
                self.Save_Parameters_and_Tuning(theta, chain_id)
                                
            if i > 50 and i % 1000 == 0 and i < self.burn_in:
                self.adapt_tuning()

    
        return (Ages_store, Initial_Th_mean_store, U234_ratios_store,
                Th232_ratios_store, Th230_ratios_store,
                posterior_store, U234_initial_store)
    

    """
    Check if files exist use pickles to start chains, else
    start the chain from random iniitalization
    """

    def check_starting_parameters(self):
        """
        Return a list of length self.n_chains of starting θ tuples.
        If Start_from_pickles is True and *all* pickle files exist,
        load them; otherwise generate new ones.
        """
        if self.Start_from_pickles:
            # Try to load all pickles
            loaded = []
            for chain_id in range(self.n_chains):
                fname = f'{self.sample_name}_theta_{chain_id}.pkl'
                if os.path.exists(fname):
                    with open(fname, "rb") as f:
                        loaded.append(pickle.load(f))
                else:
                    break  # missing one pickle → abandon loading
            if len(loaded) == self.n_chains:
                print("Loaded starting θ from pickles")
                return loaded

        # Fallback: generate fresh starting θ’s
        print("Generating new starting θ’s")
        return self.Initial_Guesses_for_Model()


    def Run_MCMC(self):
        iterations = self.iterations
        chain_ids = range(self.n_chains)
        all_thetas = self.check_starting_parameters()
        

        def run_chain(theta, chain_id):
            return self.MCMC(theta, self.iterations, chain_id)

        results = Parallel(n_jobs = -1)(delayed(run_chain)(theta,
                                                           chain_id) for
                                        theta, chain_id in zip(all_thetas[:self.n_chains],
                                                               chain_ids))

        self.Chain_Results = results

        return self.Chain_Results

    def Ensure_Chain_Results(self):
        if self.Chain_Results is None:
           self.Chain_Results = self.Run_MCMC()

        return self.Chain_Results


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

        for chain_id, result in enumerate(Results_, start = 1):
            result_dict = {}
            for var_name, value in zip(z_vars, result):
                result_dict[f"{var_name}_{chain_id}"] = value

            results_dict.append(result_dict)
            
        return results_dict

    def Get_Posterior_plot(self):

        result_dicts = self.Get_Results_Dictionary()
        log_p = []
        for i in range(1, self.n_chains + 1):
            chain_dict = result_dicts[i-1]  # get the dictionary for chain i
            log_p.append(chain_dict[f"z6_{i}"])

        fig, ax = plt.subplots(1,1, figsize = (5,5))

        for i in range(self.n_chains):
            ax.plot(log_p[i],
               label = f'Chain {i + 1}')
    
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Posterior')
        ax.legend(frameon = True, loc = 4, fontsize = 10, ncol = 2)


    def Get_Posterior_Values(self):
        result_dicts = self.Get_Results_Dictionary()
        log_p = []
        for i in range(1, self.n_chains + 1):
            chain_dict = result_dicts[i-1]  # get the dictionary for chain i
            log_p.append(chain_dict[f"z6_{i}"])

        return log_p


    def Get_Useries_Ages(self):
        # Gather each chain’s draws for the “z1” ages
        result_dicts = self.Get_Results_Dictionary()
        chains = []
        for i in range(1, self.n_chains + 1):
            chains.append(result_dicts[i-1][f"z1_{i}"])    # shape (n_samples, N_meas)
        
        # Combine into one big array of shape (n_chains*n_samples, N_meas)
        all_draws = np.vstack(chains)
        
        # Posterior means
        means = all_draws.mean(axis=0)
        
        # 95% credible interval via the 2.5th and 97.5th percentiles
        lower, upper = np.percentile(all_draws, [2.5, 97.5], axis=0)
        
        # Errors relative to the mean
        lower_err = means - lower
        upper_err = upper - means
        
        return means, [lower_err, upper_err]




    def Get_Initial_Thoriums(self):
        # Gather each chain’s draws for the “z1” ages
        result_dicts = self.Get_Results_Dictionary()
        chains = []
        for i in range(1, self.n_chains + 1):
            chains.append(result_dicts[i-1][f"z2_{i}"])    # shape (n_samples, N_meas)
        
        # Combine into one big array of shape (n_chains*n_samples, N_meas)
        all_draws = np.vstack(chains)
        
        # Posterior means
        means = all_draws.mean(axis=0)
        
        # 95% credible interval via the 2.5th and 97.5th percentiles
        lower, upper = np.percentile(all_draws, [2.5, 97.5], axis=0)
        
        # Errors relative to the mean
        lower_err = means - lower
        upper_err = upper - means
        
        return means, [lower_err, upper_err]


    def Get_234U_initial(self):
        # Gather each chain’s draws for the “z1” ages
        result_dicts = self.Get_Results_Dictionary()
        chains = []
        for i in range(1, self.n_chains + 1):
            chains.append(result_dicts[i-1][f"z7_{i}"])    # shape (n_samples, N_meas)
        
        # Combine into one big array of shape (n_chains*n_samples, N_meas)
        all_draws = np.vstack(chains)
        
        # Posterior means
        means = all_draws.mean(axis=0)
        
        # 95% credible interval via the 2.5th and 97.5th percentiles
        lower, upper = np.percentile(all_draws, [2.5, 97.5], axis=0)
        
        # Errors relative to the mean
        lower_err = means - lower
        upper_err = upper - means
        
        return means, [lower_err, upper_err]



    def Save_Initial_Thorium(self):
        Model_Ini_Thorium, Model_Ini_Thorium_err = self.Get_Initial_Thoriums()

        df_thor = pd.DataFrame({"Depth_Meas" : self.data['Depths'].values,
                              "Depth_Meas_err" : self.data['Depths_err'].values,
                              "Model_initial_th" : Model_Ini_Thorium,
                              "M_Initial_Thorium_err1" : Model_Ini_Thorium_err[0],
                              "M_Initial_Thorium_err2" : Model_Ini_Thorium_err[1]})

        df_thor.to_excel(f'{self.sample_name}_Initial_Thoriums.xlsx')


    def Save_234U_Initial(self):
        Model_Ini_Thorium, Model_Ini_Thorium_err = self.Get_234U_initial()

        df_thor = pd.DataFrame({"Depth_Meas" : self.data['Depths'].values,
                              "Depth_Meas_err" : self.data['Depths_err'].values,
                              "Model_initial_th" : Model_Ini_Thorium,
                              "M_Initial_Thorium_err1" : Model_Ini_Thorium_err[0],
                              "M_Initial_Thorium_err2" : Model_Ini_Thorium_err[1]})

        df_thor.to_excel(f'{self.sample_name}_Initial_234U.xlsx')
    
    

    def Save_Useries_Ages(self):
        U_series_ages, U_series_ages_err = self.Get_Useries_Ages()

        df_ad = pd.DataFrame({"Depth_Meas" : self.data['Depths'].values,
                              "Depth_Meas_err" : self.data['Depths_err'].values,
                              "U_ages" : U_series_ages,
                              "U_Age_low" : U_series_ages_err[0],
                              "U_Age_high" : U_series_ages_err[1]})

        df_ad.to_excel(f'{self.sample_name}_U_Series_Ages.xlsx')


    def Gelman_Rubin(self, chain_list):
        """
        Calculates the Gelman-Rubin statistic for each parameter across chains.

        Args:
        chain_list (list of np.ndarray): A list where each element is an n x p matrix of samples
                                         from one MCMC chain, with n samples and p parameters.

        Returns:
        np.ndarray: An array of Gelman-Rubin statistics for each parameter.
        """
        n_ch = len(chain_list)
        n = chain_list[0].shape[0]
        # Stack chains to simplify calculations, shape is (m, n, p)
        stacked_chains = np.stack(chain_list, axis=0)
        # Calculate means across samples within each chain (m x p)
        chain_means = np.mean(stacked_chains, axis=1)
        # Overall mean across chains for each parameter (p)
        grand_mean = np.mean(chain_means, axis=0)
        # Between-chain variance for each parameter (p)
        B = (n / (n_ch - 1)) * np.sum((chain_means - grand_mean)**2, axis=0)
        # Within-chain variances for each parameter (p)
        W = np.mean(np.var(stacked_chains, axis=1, ddof=1), axis=0)
        # Estimate the variance (p)
        var_plus = ((n - 1) / n) * W + (1 / n) * B
        # Gelman-Rubin statistic (p)
        R_hat = np.sqrt(var_plus / W)
        return R_hat


    def In_Thor_Chain_Stats(self):
        # Extract Thorium chains and calculate Gelman-Rubin statistic
        result_dicts = self.Get_Results_Dictionary()


        In_Thorium_chains = []
        for i in range(self.n_chains):
            chain_dict = result_dicts[i]
            In_Thorium_chains.append(np.array(chain_dict[f"z2_{i+1}"]))
        
        # Calculate the Gelman-Rubin statistic for Thorium
        R_hat = self.Gelman_Rubin(In_Thorium_chains)
        return R_hat


    def Useries_Age_Chain_Stats(self):
        # Extract Thorium chains and calculate Gelman-Rubin statistic
        result_dicts = self.Get_Results_Dictionary()


        Uage_chains = []
        for i in range(self.n_chains):
            chain_dict = result_dicts[i]
            Uage_chains.append(np.array(chain_dict[f"z1_{i+1}"]))
        
        # Calculate the Gelman-Rubin statistic for Thorium
        R_hat = self.Gelman_Rubin(Uage_chains)
        return R_hat
        

    def symmetric_sigma_from_errs(self, lower_err, upper_err, conf=0.95):
        """
        Convert asymmetric ± errors (relative to the mean) at a given confidence
        into a single symmetric ~1σ. Works elementwise on arrays.
        """
        z = norm.ppf(0.5*(1+conf))
        lower_err = np.asarray(lower_err)
        upper_err = np.asarray(upper_err)
        return 0.5 * (upper_err/z + lower_err/z)

    def SummaryDataFrame(self):
        U_series_ages, U_series_ages_err = self.Get_Useries_Ages()
        Model_Ini_uranium, Model_Ini_uranium_err = self.Get_234U_initial()
        Model_Ini_Thorium, Model_Ini_Thorium_err = self.Get_Initial_Thoriums()
        
        
        age_sigma = self.symmetric_sigma_from_errs(U_series_ages_err[0], U_series_ages_err[1])
        thor_sigma = self.symmetric_sigma_from_errs(Model_Ini_Thorium_err[0],                                      Model_Ini_Thorium_err[1])
        uran_sigma = self.symmetric_sigma_from_errs(Model_Ini_uranium_err[0],                                     Model_Ini_uranium_err[1])
        
        df_all = pd.DataFrame({"Depth_Meas" : self.data['Depths'].values,
                              "Depth_Meas_err" : self.data['Depths_err'].values,
                              "age" : U_series_ages,
                              "age_err": age_sigma,
                              "initial thorium":Model_Ini_Thorium,
                              "initial thorium err": thor_sigma,
                              "initial uranium": Model_Ini_uranium,
                              "initial uranium err": uran_sigma})

        df_all.to_csv(f'{self.sample_name}_ibis_summary.csv')
