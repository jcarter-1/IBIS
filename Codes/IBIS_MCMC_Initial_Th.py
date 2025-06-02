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
from scipy.stats import ks_2samp, lognorm


class IBIS_MCMC:
    def __init__(self, Thor_KDE, Age_Maximum,
                 Age_Uncertainties, data, sample_name = 'SAMPLE_NAME',
                  results_folder = None,
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
            self.Thor_KDE = uniform(0, 200)
        self.burn_in = burn_in
        self.Age_Maximum = Age_Maximum
        self.Age_Uncertainties = Age_Uncertainties # 1 sigma uncertainties
        self.Th230_lam_Cheng = 9.1577e-06
        self.Th230_lam_Cheng_err = 1.3914e-08
        self.U234_lam_Cheng = 2.8263e-06
        self.U234_lam_Cheng_err = 2.8234e-09
        self.N_meas = data.shape[0]
        self.n_chains = n_chains
        self.Depths = data['Depths'].values
        self.Depths_err = data['Depths_err'].values
        self.iterations = iterations
        self.sample_name = sample_name
        self.Chain_Results = None
        self.Start_from_pickles = Start_from_pickles
        self.results_folder = results_folder
        self.depths = self.data['Depths'].values
        
        # Short Hand ratios
        self.r08 = data['Th230_238U_ratios'].values
        self.r28 = data['Th232_238U_ratios'].values
        self.r48 = data['U234_U238_ratios'].values
        # Short Hand uncertainties
        self.r08_err = data['Th230_238U_ratios_err'].values
        self.r28_err = data['Th232_238U_ratios_err'].values
        self.r48_err = data['U234_U238_ratios_err'].values
        
        
        # Combined age uncertainties at each depth
        uncertainties_squared_sum = np.square(self.Age_Uncertainties[:-1]) + np.square(self.Age_Uncertainties[1:])
        # Calculate the combined uncertainties for the difference in ages
        combined_uncertainties = np.sqrt(uncertainties_squared_sum)
        self.age_uncertainties_comb = combined_uncertainties
        self.rng = np.random.default_rng()           # <--- Helper shortcut

        # set up prior
        self.set_up_thor_prior()


        # Set up Covariances and keys and updates
        self.Sigma_230 = np.diag(self.r08_err ** 2)
        self.Sigma_232 = np.diag(self.r28_err ** 2)
        self.Sigma_234 = np.diag(self.r48_err ** 2)
        from scipy.linalg import block_diag

        self.Sigma_Measurements = block_diag(
        self.Sigma_230,
        self.Sigma_232,
        self.Sigma_234
        )
                                            
        self.Sigma_Initial_Thor = np.diag(np.ones(self.N_meas))
                                            
        self.keys = ['Measured_Ratios',
                     'Initial_Thorium']
                     
        self._move_funcs = [('Measured_Ratios' , self.Measured_Ratios_Move),
                            ('Initial_Thorium' , self.Initial_Thorium_Move)]
                            
        
        
    def stabilize_cov(self, cov, min_eig=1e-8, diag_jitter=1e-8):
        """
        Given a (d×d) covariance matrix `cov`, force it to be symmetric,
        clip any eigenvalues < min_eig, and add diag_jitter along the diagonal.
        Ensures positive definiteness before calling multivariate_normal().
        need to ensure positive definitiness before calling mutivariate_normal
        """
        # 1) Symmetrize
        cov_sym = 0.5 * (cov + cov.T)
        # 2) Eigen‐decompose
        eigvals, eigvecs = np.linalg.eigh(cov_sym)
        # 3) Clip eigenvalues
        eigvals_clipped = np.clip(eigvals, a_min=min_eig, a_max=None)
        # 4) Reconstruct a PD matrix
        cov_psd = (eigvecs * eigvals_clipped) @ eigvecs.T
        # 5) Add tiny jitter on the diagonal
        cov_psd += np.eye(cov_psd.shape[0]) * diag_jitter
        return cov_psd
                            
                        
    # Set up for thor_cdf

    def set_up_thor_prior(self):
        # 1) pick a grid spanning your support
        x_min, x_max = 0, 500
        N = 2000
        x_grid = np.linspace(x_min, x_max, N)

        # 2) evaluate the PDF on that grid
        pdf_grid = np.maximum(self.Thor_KDE(x_grid), 0.0)

        # 3) do a cumulative trapezoidal integral to get un‐normalized CDF
        dx = x_grid[1] - x_grid[0]
        cdf_vals = np.cumsum((pdf_grid[:-1] + pdf_grid[1:]) * 0.5) * dx

        # 4) tack on the last point and normalize to [0,1]
        cdf_vals = np.concatenate(([0.0], cdf_vals))
        cdf_vals /= cdf_vals[-1]

        # 5) build your CDF interpolant
        self._thor_cdf = interp1d(x_grid, cdf_vals,
                                  kind='linear',
                                  bounds_error=False,
                                  fill_value=(0.0, 1.0))
        self._thor_ppf = interp1d(cdf_vals, x_grid, fill_value=(x_min,x_max), bounds_error=False)

     

    """
    Age Equation Stuff
    """
    def U_series_age_equation(self,
                              age,  Th_230_lam,
                              U_234_lam, Th_initial, Th232_238U_ratio,
                              U234_U238_ratio, Th230_U238_ratio):
        """
        Internal method to compute the left and right sides of the U-series age equation.
        """
        dlam = Th_230_lam - U_234_lam
        LEFT_SIDE = Th230_U238_ratio
        LEFT_SIDE_2 =  Th232_238U_ratio * Th_initial * np.exp(-Th_230_lam * age)
        RIGHT_SIDE_1 =  1-np.exp(-Th_230_lam * age)
        RIGHT_SIDE_2 = ((U234_U238_ratio) - 1) * (Th_230_lam /dlam) * (1 - np.exp(-dlam * age))
        
        return LEFT_SIDE - LEFT_SIDE_2 - RIGHT_SIDE_1 - RIGHT_SIDE_2


    def calculate_age_solver(self, Th_230_lam, U_234_lam, Th_initial,
                             Th232_238U_ratio, U234_U238_ratio,
                             Th230_238U_ratio, age_guess=200):
        """
        Solve the U-series age equation for a single age estimate.
        """
        func = lambda age: self.U_series_age_equation(age, Th_230_lam, U_234_lam,
                                                       Th_initial, Th232_238U_ratio, U234_U238_ratio, Th230_238U_ratio)
        t_solution = fsolve(func, age_guess)
        return t_solution[0]

    def u_series_ages_(self, Th_230_lam, U_234_lam,
                       Th_initials,
                       Th232_238U_ratios, U234_U238_ratios,
                       Th230_238U_ratios, age_guess=200):
        """
        Calculate U-series ages for multiple sets of initial conditions using a vectorized approach.
        """
        Useries_ages = np.array([
            self.calculate_age_solver(Th_230_lam, U_234_lam, Th_initial,
                                      Th232_238U_ratio, U234_U238_ratio, Th230_238U_ratio, age_guess)
            for Th_initial, Th232_238U_ratio, U234_U238_ratio, Th230_238U_ratio in zip(Th_initials, Th232_238U_ratios, U234_U238_ratios, Th230_238U_ratios)])
        return Useries_ages
    
    def u_series_ages_initialize(self, Th_230_lam, U_234_lam,
                                 Th_initial, Th232_238U_ratios,
                                 U234_U238_ratios, Th230_238U_ratios,
                                 age_guess=200):
        """
        Calculate U-series ages for arrays of initial conditions using an iterative approach.
        """
        calculated_ages = np.zeros(len(Th232_238U_ratios))  # Initialize an array to store the calculated ages
        for i in range(len(Th232_238U_ratios)):
            calculated_ages[i] = self.calculate_age_solver(Th_230_lam, U_234_lam, Th_initial,
                                                           Th232_238U_ratios[i],     U234_U238_ratios[i],
                                                           Th230_238U_ratios[i], age_guess)
        return calculated_ages


    def check_bounds(self, values, bounds):
        """ Check if each value in the array is within its corresponding bounds.
        Args:
            values (np.array): An array of values to check.
            bounds (np.array): An array of (min, max) bounds for each value.
    
        Returns:
            float: 0 if all values are within their bounds, -np.inf otherwise.
        """
        if np.all((values >= bounds[:, 0]) & (values <= bounds[:, 1])):
            return 0
        else:
            return -np.inf
            

    def Initial_Thorium_Prior(self, Initial_Thorium):


        lp = np.sum(np.log(self.Thor_KDE(Initial_Thorium)))
        
        return lp
            
    def Improper_Age_Prior(self, x):
        """
        Heavy penalty (−1e10) plus a tiny exp(−distance), floored at 1e-50,
        for any xi < 0 or xi > Age_Maximum.
        """
        lp = 0.0
        for xi in x:
            if xi < 0 or xi > self.Age_Maximum:
                # big “improper” penalty
                lp -= 1e10
    
                # distance to nearest boundary
                if xi < 0:
                    delta = -xi
                else:
                    delta = xi - self.Age_Maximum
    
                # small extra penalty, but never below 1e-50
                penalty = np.clip(np.exp(-delta), 1e-50, None)
                lp      -= penalty
    
        return lp

    def Ln_Priors(self, theta):
        # Unpack Theta
        Initial_Th_mean, U234_ratios, \
        Th230_ratios, Th232_ratios = theta

        # Calculate U-Series Ages (Placeholder for actual age calculation)
        USeries_Ages = self.u_series_ages_(self.Th230_lam_Cheng,
                                    self.U234_lam_Cheng, Initial_Th_mean,
                                    Th232_ratios, U234_ratios, Th230_ratios)


        """
        Start Priors
        """
        lp = 0

        if np.any(Initial_Th_mean < 0):
            return - np.inf

        """
        Initial Thorium
        """
        lp += self.Initial_Thorium_Prior(Initial_Th_mean)

        """
        Age Prior
        ---------
        - Age must be between 0 and the maximum Age
        """
        lp += self.Improper_Age_Prior(USeries_Ages)

        """
        Measurement Ratios
        """
        lp += np.sum(norm.logpdf(U234_ratios, self.data['U234_U238_ratios'].values,
                                    self.data['U234_U238_ratios_err'].values))
    
        lp += np.sum(norm.logpdf(Th230_ratios, self.data['Th230_238U_ratios'].values,
                                 self.data['Th230_238U_ratios_err'].values))
            
        lp += np.sum(norm.logpdf(Th232_ratios, self.data['Th232_238U_ratios'].values,
                             self.data['Th232_238U_ratios_err'].values))

        return lp


    def Initial_Thorium_sampler(self):
        xmin = 0
        xmax = 500
        grid_points = 1000
        x_grid = np.linspace(xmin, xmax, grid_points)
        pdf_vals = self.Thor_KDE(x_grid)
    
        # Compute the cumulative distribution function (CDF) using the trapezoidal rule.
        cdf_vals = np.cumsum(pdf_vals)
        # Normalize the CDF
        cdf_vals = cdf_vals / cdf_vals[-1]
    
        #Build an inverse CDF (quantile function) interpolator.
        inv_cdf = interp1d(cdf_vals,
        x_grid, bounds_error=False,
        fill_value=(xmin, xmax))
    
        # Generate uniform random samples between 0 and 1, and invert them.
        u = np.random.rand(self.N_meas)
        samples = inv_cdf(u)
        Initial_Thorium_guess = samples
        
        return abs(Initial_Thorium_guess)
        

    def Initial_Guesses_for_Model(self, max_attempts=1000):
        """
        Draw self.n_chains valid initial thetas by rejection sampling on the age‐constraints.
        If we hit max_attempts without finding a fully valid candidate, we accept the
        last one (so the chain can still start) but it will carry the huge improper‐age prior.
        """
        initial_thetas = []
    
        for chain in range(self.n_chains):
            attempts = 0
            last_theta = None
    
            while attempts < max_attempts:
                attempts += 1
    
                # 1) propose everything
                Initial_Th = self.Initial_Thorium_sampler()
    
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
    
                theta = (Initial_Th, U234_in, Th230_in, Th232_in)
                # 2) compute ages for this candidate
                ages = self.U_series_ages_theta(theta)
    
                # 3) reject early if any age is out of bounds
                if np.any(ages < 0) or np.any(ages > self.Age_Maximum):
                    last_theta = (Initial_Th, U234_in, Th230_in, Th232_in)
                    continue
    
                # 4) if ages are OK, compute the true prior
                logp  = self.Ln_Priors(theta)
                if logp > -np.inf:
                    initial_thetas.append(theta)
                    break
                else:
                    last_theta = theta
    
            else:
                # fallback: we never found a fully valid one in max_attempts
                # append last_theta anyway (it will carry a huge penalty in Ln_Priors)
                initial_thetas.append(last_theta)
    
        return initial_thetas
            
            
    def Strat_LogLikelihood(self, ages):
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


    def Log_Likelihood(self, theta):
        # Unpack Theta
        Initial_Th_mean, U234_ratios, \
        Th230_ratios, Th232_ratios  = theta

        Ages_ = self.u_series_ages_(self.Th230_lam_Cheng,
                                    self.U234_lam_Cheng,
                                    Initial_Th_mean, Th232_ratios,
                                     U234_ratios, Th230_ratios,
                                        age_guess = self.Age_Maximum/2)

        Strat_LogLike = self.Strat_LogLikelihood(Ages_)

        return Strat_LogLike

    # Posterior
    def Log_Posterior(self, theta):
        """
        Function for determining Posterior
        ----------------------------------
        - This is Bayes-Price-Laplace Theorem
        in proportional and log form
        """
        lp = self.Ln_Priors(theta)
        if not np.isfinite(lp):
            return - np.inf
        else:
            return lp + self.Log_Likelihood(theta)

            
            
    ##############################################
    # ============= Moves for MCMC ============= #
    ##############################################
    def Measured_Ratios_Move(self, theta, tuning, Sigma):
        """
        Log-normal RW for one row of C_Kr
        """
        init_th, U234, Th230, Th232 = theta
        logp_cur = self.Log_Posterior(theta)

        # log-transform row
        Measurements = np.vstack([Th230, Th232, U234])
        
                # … inside Measured_Ratios_Move(…) …
        K, M = Measurements.shape
        logC = np.log(Measurements).ravel()

        # ─── Build a “stabilized” covariance ───
        cov_emp = Sigma * (tuning**2)
        cov_emp = np.array(cov_emp, order='F')
        cov_pd = self.stabilize_cov(cov_emp)         # <--- NEW

        try:
            delta = self.rng.multivariate_normal(
                np.zeros(K*M), cov_pd
            )
        except np.linalg.LinAlgError:
            # Fallback: independent normals if PD check still fails
            delta = self.rng.normal(0, tuning, size=K*M)

        logC_prop = logC + delta
        
        self.sample_histories['Measured_Ratios'].append(logC_prop.copy())

        # 3) back to positive space & reshape
        C_prop = np.exp(logC_prop).reshape(K, M)
        
        # reject immediately if any < 0 (shouldn’t happen after exp, but just in case)
        if np.any(C_prop <= 0):
            return (U234, Th230, Th232), False

        
        Th230_prop = C_prop[0,:]
        Th232_prop = C_prop[1,:]
        U234_prop = C_prop[2, :]
        
        # 4) MH test
        theta_p   = (init_th, U234_prop, Th230_prop, Th232_prop)
        logp_prop = self.Log_Posterior(theta_p)
        if np.isfinite(logp_prop) and np.log(np.random.rand()) < (logp_prop - logp_cur):
            # we accept; store the new log‐ratios logC_prop
            return (U234_prop, Th230_prop, Th232_prop), True
        else:
            # reject; store the old log‐ratios logC
            return (U234, Th230, Th232), False
            
    
            
    def Initial_Thorium_Move(self, theta, tuning, Sigma):
        init_th, U234, Th230, Th232 = theta
        logp_cur = self.Log_Posterior(theta)
    
        if np.random.rand() < 0.05:
            init_th_prop = self.Initial_Thorium_sampler()
        else:
            Log_InT = np.log(init_th)
            cov = Sigma * (tuning**2)
            try:
                delta = np.random.multivariate_normal(
                    np.zeros_like(Log_InT), cov
                )
            except np.linalg.LinAlgError:
                delta = np.random.normal(0, tuning, size=Log_InT.shape)
            log_init_th_prop = Log_InT + delta
            init_th_prop = np.exp(log_init_th_prop)
            
            if np.any(init_th_prop < 0):
                # immediately reject, but still append the *old* vector
                return init_th, False
    
        theta_p   = (init_th_prop, U234, Th230, Th232)
        logp_prop = self.Log_Posterior(theta_p)
        self.sample_histories['Initial_Thorium'].append(init_th_prop.copy())

        # Always append whichever vector we are “currently holding”:
        if np.isfinite(logp_prop) and np.log(np.random.rand()) < (logp_prop - logp_cur):
            # accept: store the new proposal
            return init_th_prop, True
        else:
            # reject: store the old one
            return init_th, False
                

    def update_vector_parameters(self, theta, new_value, move_name, index):
        """
        Helper function for updating the vector values within the MCMC function
        """
        # Unpack Theta
        Initial_Th_mean, U234_ratios, \
        Th230_ratios, Th232_ratios  = theta
        
        if move_name == 'Initial_Th_mean_Z':
            Initial_Th_mean[index] = new_value

        return (Initial_Th_mean, U234_ratios, \
        Th230_ratios, Th232_ratios)


    def U_series_ages_theta(self, theta):
        # Unpack Theta
        Initial_Th_mean, U234_ratios, \
        Th230_ratios, Th232_ratios = theta
        
        Useries_ages = self.u_series_ages_(self.Th230_lam_Cheng,
                                            self.U234_lam_Cheng,
                                           Initial_Th_mean, Th232_ratios,
                                           U234_ratios, Th230_ratios,
                                           age_guess = 300)
        return Useries_ages


    def Initial_234U(self, theta):
        Initial_Th_mean, U234_ratios, \
        Th230_ratios, Th232_ratios = theta

        Uages = self.U_series_ages_theta(theta)
        

        U234_initial  = 1 + ((U234_ratios - 1) * np.exp(self.U234_lam_Cheng * Uages))
        
        return U234_initial


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
                self.tuning_factors = pickle.load(f)
        else:
            self.tuning_factors = {}
            self.tuning_factors['Measured_Ratios'] = 1
            self.tuning_factors['Initial_Thorium'] = 1
            
        Sigma_Prop_file = f'Sigma_{self.sample_name}_{chain_id}.pkl'
        if os.path.exists(Sigma_Prop_file) and self.Start_from_pickles:
            with open(Sigma_Prop_file, 'rb') as f:
                self.Sigma_proposals = pickle.load(f)
        else:
                self.Sigma_proposals = {}
                self.Sigma_proposals['Measured_Ratios'] = self.Sigma_Measurements
                self.Sigma_proposals['Initial_Thorium'] = self.Sigma_Initial_Thor
                
        # 3) init counters & histories
        self.proposal_counts = {k:0 for k in self.keys}
        self.accept_counts   = {k:0 for k in self.keys}
        self.sample_histories = {k:[] for k in self.keys if k in self.Sigma_proposals}
    
        # Initial model ages
        model_ages = self.u_series_ages_(self.Th230_lam_Cheng,
                                         self.U234_lam_Cheng,
                                         Initial_Th_mean,
                                         Th232_ratios, U234_ratios,
                                         Th230_ratios, age_guess=self.Age_Maximum/2)
    
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

            if move_name == 'Measured_Ratios':
                In_Th, U234, Th230, Th232 = theta
                key= 'Measured_Ratios'
                self.proposal_counts[key] += 1
                
                new_values, accept = move_func(theta, self.tuning_factors[key], self.Sigma_proposals[key])
                if accept:
                    self.accept_counts[key] += 1
                    theta = (In_Th, new_values[0], new_values[1], new_values[2])
                    current_block = np.array([theta[1], theta[2], theta[3]])

    
            if move_name in ['Initial_Thorium']:
                In_Th, U234, Th230, Th232 = theta
                key= 'Initial_Thorium'
                self.proposal_counts[key] += 1
                
                new_values, accept = move_func(theta, self.tuning_factors[key], self.Sigma_proposals[key])
                if accept:
                    self.accept_counts[key] += 1
                    theta = (new_values, U234, Th230, Th232)
                    
            # Store samples only after burn-in
            if i > self.burn_in:
                # Save the current state to our storage arrays
                Ages_store[sample_index, :] = self.U_series_ages_theta(theta)
                Initial_Th_mean_store[sample_index, :] = theta[0]
                U234_initial_store[sample_index, :] = self.Initial_234U(theta)
                U234_ratios_store[sample_index, :] = theta[1]
                Th232_ratios_store[sample_index, :] = theta[2]
                Th230_ratios_store[sample_index, :] = theta[3]
                posterior_store[sample_index] = self.Log_Posterior(theta)
                sample_index += 1
                
            """
            Chain Behaviour and Updates here
            """
            if i > 50 and i % 1000 == 0:
                # 1) save the current theta
                with open(f'{self.sample_name}_theta_{chain_id}.pkl', 'wb') as f:
                    pickle.dump(theta, f)

                # 2) save tuning_factors
                with open(tf_file, 'wb') as f:
                    pickle.dump(self.tuning_factors, f)

                # 3) make sure none of the Sigma_proposals arrays is a memmap,
                #    then pickle the dict itself:
                for key in self.Sigma_proposals:
                    self.Sigma_proposals[key] = np.array(self.Sigma_proposals[key], order='F')

                with open(Sigma_Prop_file, 'wb') as f:
                    pickle.dump(self.Sigma_proposals, f)
                                
                    
            if i > 50 and i % 1000 == 0 and i < self.burn_in:
                for param, history in self.sample_histories.items():
                    if len(history) == 0:
                        continue
                    H = np.vstack(history)
                    if H.shape[0] > (len(H[0] + 1)):
                        Sigma_emp = np.cov(H, rowvar = False)
                        d = Sigma_emp.shape[0]
                        self.Sigma_proposals[param]= Sigma_emp + 1e-8*np.eye(d)
                    history.clear()
                for param in self.tuning_factors.keys():
                    if self.proposal_counts[param] ==0:
                        continue
                    a = self.accept_counts[param]
                    p = self.proposal_counts[param]
                    
                    block_accept_rate = a / p
                    
                    if block_accept_rate < 0.234:
                        self.tuning_factors[param] *= 0.9
                    else:
                        self.tuning_factors[param] *= 1.1
                self.accept_counts[param] = 0.0
                self.proposal_counts[param] = 0.0
    
        return (Ages_store, Initial_Th_mean_store, U234_ratios_store,
                Th232_ratios_store, Th230_ratios_store, posterior_store, U234_initial_store)
    

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
