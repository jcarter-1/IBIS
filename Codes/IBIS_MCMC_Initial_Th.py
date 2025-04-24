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
                 Th230_lam_Cheng = 9.1577e-06, 
                 Th230_lam_Cheng_err = 1.3914e-08,
                 U234_lam_Cheng = 2.8263e-06, 
                 U234_lam_Cheng_err= 2.8234e-09, 
                 n_chains = 3, 
                 iterations = 50000,
                 burn_in = 10000,
                 Start_from_pickles = True, 
                 Include_Detrital = None): 
        
        self.data = data
        self.Thor_KDE = Thor_KDE
        if self.Thor_KDE is None:
            # If there is no prior
            # We will assume a uniform prior
            # Th ~U(0, 200)
            self.Thor_KDE = uniform(0, 200)
        self.burn_in = burn_in
        self.Age_Maximum = Age_Maximum
        self.Age_Uncertainties = Age_Uncertainties # 2 sigma uncertainties
        self.Th230_lam_Cheng = Th230_lam_Cheng
        self.Th230_lam_Cheng_err = Th230_lam_Cheng_err
        self.U234_lam_Cheng = U234_lam_Cheng
        self.U234_lam_Cheng_err = U234_lam_Cheng_err
        self.N_meas = data.shape[0]
        self.n_chains = n_chains
        self.Depths = data['Depths'].values
        self.Depths_err = data['Depths_err'].values
        self.iterations = iterations
        self.sample_name = sample_name
        self.Chain_Results = None
        self.Start_from_pickles = Start_from_pickles
        self.Include_Detrital = Include_Detrital
        
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
                
    """
    Ratio Covariance full correlation
    """
    def Ratio_Cov_full_corr(self, index):
        """
        Build a 3×3 covariance matrix for
        [Th230/U238, Th232/U238, U234/U238] at sample `index`,
        assuming 100% correlation via the common U238 error.
        Maximum uncertainty here
        """
        # pull the three 1σ ratio errors at this sample
        σ230 = self.r08_err[index]
        σ232 = self.r28_err[index]
        σ234 = self.r48_err[index]
        errs = np.array([σ230, σ232, σ234])      # shape (3,)
        Σr = np.outer(errs, errs)               # shape (3,3)
        return Σr
     

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
        LEFT_SIDE = Th230_U238_ratio
        LEFT_SIDE_2 =  Th232_238U_ratio * Th_initial * np.exp(-Th_230_lam * age)
        RIGHT_SIDE_1 = 1 - np.exp(-Th_230_lam * age)
        RIGHT_SIDE_2 = ((U234_U238_ratio) - 1) * (Th_230_lam / (U_234_lam - Th_230_lam)) * (1 - np.exp((U_234_lam - Th_230_lam) * age))
        
        return LEFT_SIDE - LEFT_SIDE_2 - RIGHT_SIDE_1 + RIGHT_SIDE_2


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
            calculated_ages[i] = self.calculate_age_solver(Th_230_lam, U_234_lam,                                               Th_initial,
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
        lp = 0
        for i in range(len(x)):
            if x[i] <0 or x[i] > self.Age_Maximum:
                lp += -1e30
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
        xmax = 50
        grid_points = 128
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
        


    def Initial_Guesses_for_Model(self): 

        initial_thetas = []

        for i in range(self.n_chains): 
            log_prior = - np.inf
            while log_prior == - np.inf:


                Initial_Th = self.Initial_Thorium_sampler()

                Th230_ratios_in = np.random.normal(self.data['Th230_238U_ratios'].values,
                                                   self.data['Th230_238U_ratios_err'].values)
                
                Th232_ratios_in = np.random.normal(self.data['Th232_238U_ratios'].values,
                                                   self.data['Th232_238U_ratios_err'].values)
                        
                U234_ratios_in = np.random.normal(self.data['U234_U238_ratios'].values, 
                                                  self.data['U234_U238_ratios_err'].values)
        
                
                Useries_ages =  self.u_series_ages_(self.Th230_lam_Cheng,
                                                self.U234_lam_Cheng,
                                               Initial_Th, 
                                               Th232_ratios_in,  
                                               U234_ratios_in, 
                                               Th230_ratios_in, 
                                               age_guess=self.Age_Maximum/2)
        
                theta_initial = Initial_Th, \
                U234_ratios_in, \
                Th230_ratios_in, Th232_ratios_in
        
                        
                log_prior = self.Ln_Priors(theta_initial) 
        
                if log_prior != -np.inf: 
                    
                    initial_thetas.append(theta_initial)

        return initial_thetas
        
        
    def Strat_LogLikelihood(self,ages):
        """
        A Likelihood function to maximize the information
        from stratigraphic order accounting for measured uncertainties
        --------------------------------------------------------------
        """
        LL = 0.0
        n = len(ages)
        zeta = 3
        # Loop through each age
        for i in range(n):
            # Compare with every subsequent age
            for j in range(i+1, n):
                # Calculate the age difference
                diff = ages[j] - ages[i]
                combined_unc = np.sqrt(self.Age_Uncertainties[i]**2 + self.Age_Uncertainties[j]**2)
                if diff > 0:
                    LL += 0
                    
                elif abs(diff) > combined_unc:
                    #Use combined uncertainty of upper age and lower age
                    # Compute the probability using the normal log CDF
                    prob = norm.logcdf(diff / (combined_unc))
                    LL += prob
                else:
                    prob = norm.logcdf(diff / (zeta * combined_unc))
                    LL += prob
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
    def Measured_Ratio_Move(self, theta, tuning_factor, index):
        Initial_Th_mean, U234_ratios, \
        Th230_ratios, Th232_ratios = theta
        cov_index = self.Ratio_Cov_full_corr(index)
        current = np.array([U234_ratios[index], Th230_ratios[index], Th232_ratios[index]])
        step = np.random.multivariate_normal(mean=np.zeros(3), cov = cov_index)
        proposed = current + step*tuning_factor
        
        if np.any(proposed < 0):
            return (U234_ratios[index], Th230_ratios[index], Th232_ratios[index]), False
            
        # 5) assemble theta'
        U234_p, Th230_p, Th232_p = U234_ratios.copy(), Th230_ratios.copy(), Th232_ratios.copy()
        U234_p[index], Th230_p[index], Th232_p[index] = proposed
        theta_p = (Initial_Th_mean, U234_p, Th230_p, Th232_p)

        # 6) acceptance as before
        ll_cur, ll_prop = self.Log_Likelihood(theta), self.Log_Likelihood(theta_p)
        pr_cur, pr_prop = self.Ln_Priors(theta),    self.Ln_Priors(theta_p)
        
        if np.isneginf(ll_prop) or np.isneginf(pr_prop):
            return (U234_ratios[index], Th230_ratios[index], Th232_ratios[index]), False

        loga = (ll_prop+pr_prop) - (ll_cur+pr_cur)
        if np.log(np.random.rand()) < loga:
            return (U234_p[index], Th230_p[index], Th232_p[index]), True
        else:
            return (U234_ratios[index], Th230_ratios[index], Th232_ratios[index]), False
            
    
    def Initial_Thorium_move(self, theta, tuning_factor, index):
        # Unpack Theta
        Initial_Th_mean, U234_ratios, \
        Th230_ratios, Th232_ratios = theta

        # Move
        Initial_Th_mean_prime = np.copy(Initial_Th_mean)
        step_size = np.random.normal(0,
                                    tuning_factor)
        Initial_Th_mean_prime[index] =Initial_Th_mean_prime[index] + step_size

        # Check
        if np.any(Initial_Th_mean_prime < 0):
            return Initial_Th_mean, False

        # Update Theta
        theta_prime = Initial_Th_mean_prime, U234_ratios, \
                    Th230_ratios, Th232_ratios
        # Acceptance
        Loglike_current = self.Log_Likelihood(theta)
        Loglike_proposed = self.Log_Likelihood(theta_prime)
        Prior_current = self.Ln_Priors(theta)
        Prior_proposed = self.Ln_Priors(theta_prime)

        if Prior_proposed == -np.inf or Loglike_proposed ==-np.inf:
            return Initial_Th_mean, False

        Prior_ratio = Prior_proposed - Prior_current

        Logpost_proposed = Loglike_proposed  + Prior_proposed
        Logpost_current = Loglike_current  + Prior_current

        u = np.random.rand()
        if  u < np.exp(Loglike_proposed + Prior_ratio -Loglike_current):
            return Initial_Th_mean_prime, True

        else:
            return Initial_Th_mean, False
            

    def Logphi_move(self, theta, tuning):
        # unpack
        InitTh, U234, Th230, Th232, logφ = theta
        proposal = logφ + np.random.normal(0, tuning)
        theta_p = (InitTh, U234, Th230, Th232, proposal)
        lp, ll = self.Ln_Priors(theta), self.Log_Likelihood(theta)
        lp_p, ll_p = self.Ln_Priors(theta_p), self.Log_Likelihood(theta_p)
        if np.log(np.random.rand()) < (lp_p+ll_p)-(lp+ll):
            return proposal, True
        else:
            return logφ, False


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
    
        # Tuning Factors File Path
        tuning_factors_file = f'tuning_factor_{self.sample_name}_{chain_id}.pkl'
        if os.path.exists(tuning_factors_file) and self.Start_from_pickles:
            with open(tuning_factors_file, 'rb') as f:
                tuning_factors = pickle.load(f)
        else:
            # If no file exists then initialize the tuning factors
            tuning_factors = {}
            for i in range(Ndata):
                tuning_factors[f'Measured_Ratios_Z_{i}'] =  0.05
                tuning_factors[f'Initial_Th_mean_Z_{i}'] = 0.5
    
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
    
        # Store counters for proposals and acceptances per parameter
        proposal_counts = {p: 0 for p in tuning_factors}
        accept_counts   = {p: 0 for p in tuning_factors}
        ema_accept_rate = {p: 0.0 for p in tuning_factors}
        ema_alpha = 0.075  # smoothing factor
    
        def adaptation_step(i):
            """Return a small but decreasing step size."""
            return 0.01  # Alternatively: 1.0 / np.sqrt(i+1)
    
        target_accept_rate = 0.234
    
        move_functions = [
            ('Initial_Th_mean_Z', self.Initial_Thorium_move),
            ('Measured_Ratios_Z', self.Measured_Ratio_Move)
        ]
    
        sample_index = 0  # Counter for samples after burn-in
    
        for i in range(1, total_iterations + 1):
            move_name, move_func = random.choice(move_functions)
            
            if move_name == 'Measured_Ratios_Z':
                idx = np.random.randint(0, Ndata)
                key = f'{move_name}_{idx}'
                proposal_counts[key] += 1
                # --- do the block proposal
                new_values, accepted = move_func(
                    theta, tuning_factors[key], idx)
                if accepted:
                    accept_counts[key] += 1
                    Th0, U234, Th230, Th232 = theta
                    U234[idx], Th230[idx], Th232[idx] = new_values
                    theta = (Th0, U234, Th230, Th232)
    
            elif move_name in ['Initial_Th_mean_Z']:
                index = np.random.randint(0, Ndata)
                specific_counter_name = f'{move_name}_{index}'
                proposal_counts[specific_counter_name] += 1
                new_value, accepted = move_func(theta,
                                                tuning_factors[f'{move_name}_{index}'],
                                                index)
                if accepted:
                    accept_counts[specific_counter_name] += 1
                    new_theta = self.update_vector_parameters(theta,
                                                              new_value[index],  #  update single value
                                                              move_name,
                                                              index)
                    theta = new_theta
    
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
    
            # ---------------------------------------------------------
            #  Periodic adaptation of the tuning factors
            # ---------------------------------------------------------
            if i > 50 and i < self.burn_in and i % 5000 == 0:

                for param in tuning_factors.keys():
                    if proposal_counts[param] == 0:
                        continue
                    a = accept_counts[param]
                    p = proposal_counts[param]
                    print(f"{param}: accept {a}/{p} = {a/p:.2f}")
                    
                    block_accept_rate = accept_counts[param] / proposal_counts[param]
                    print(block_accept_rate)
                    if block_accept_rate < target_accept_rate:
                        tuning_factors[param] *= 0.9
                    else:
                        tuning_factors[param] *= 1.1
                    accept_counts[param] = 0
                    proposal_counts[param] = 0
    
            # ---------------------------------------------------------
            #  Periodically pickle the chain and tuning factors
            # ---------------------------------------------------------
            if (i + 1) % 5000 == 0:
                with open(f'{self.sample_name}_theta_{chain_id}.pkl', 'wb') as f:
                    pickle.dump(theta, f)
                with open(f'tuning_factor_{self.sample_name}_{chain_id}.pkl', 'wb') as f:
                    pickle.dump(tuning_factors, f)
    
        return (Ages_store, Initial_Th_mean_store, U234_ratios_store,
                Th232_ratios_store, Th230_ratios_store, posterior_store,    U234_initial_store)
    

    """
    Check if files exist use pickles to start chains, else
    start the chain from random iniitalization
    """
    def check_starting_parameters(self):
        thetas = []
        for chain_id in range(self.n_chains):
            theta_pickle_file = f'{self.sample_name}_theta_{chain_id}.pkl'
            if os.path.exists(theta_pickle_file) and self.Start_from_pickles:
                print(f'Pickles file exists and Start_from_pickles = {self.Start_from_pickles}')
                with open(theta_pickle_file, 'rb') as f:
                    theta_p = pickle.load(f)
                thetas.append(theta_p)
            else:
                # If no initial file exists, use initial guesses
                if not thetas or self.Start_from_pickles:  # Ensure it's only done once if needed
                    thetas = self.Initial_Guesses_for_Model()
                    
        return thetas


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


    def u234_initial_Chain_Stats(self):
        # Extract Thorium chains and calculate Gelman-Rubin statistic
        result_dicts = self.Get_Results_Dictionary()


        lam234_chains = []
        for i in range(self.n_chains):
            chain_dict = result_dicts[i]
            lam234_chains.append(np.array(chain_dict[f"z7_{i+1}"]))
        
        # Calculate the Gelman-Rubin statistic for Thorium
        R_hat = self.Gelman_Rubin(lam234_chains)
        return R_hat
