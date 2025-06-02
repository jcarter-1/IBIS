import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, uniform, lognorm
from scipy.interpolate import interp1d
import matplotlib as mpl
from tqdm import tqdm, tnrange, tqdm_notebook
import dill as pickle
import time
import os
from joblib import Parallel, delayed
import random



class IBIS_Strat:
    def __init__(self, U_series_ages,
                 U_series_ages_err_low, U_series_ages_err_high,
                  data, sample_name = 'SAMPLE_NAME',
                 Start_from_pickles = True, n_chains = 3,
                 iterations = 50000, burn_in = 10000,
                Top_Age_Stal = False):
        

        self.Start_from_pickles = Start_from_pickles
        self.n_chains = n_chains
        self.iterations = iterations
        self.burn_in = burn_in
        self.sample_name = sample_name
        self.Top_Age_Stal = Top_Age_Stal

        if self.Top_Age_Stal:
            U_age_top = 0
            U_age_top_err = 0.1
            

        if self.Top_Age_Stal:
            self.u_ages = np.insert(U_series_ages, 0,  U_age_top)
            self.low_age_err = np.insert(U_series_ages_err_low, 0, 0) / 2 # Data is read in at the 95% confidence
            self.high_age_err = np.insert(U_series_ages_err_high, 0, 0.1) / 2 # Data is read in at the 95% confidence
            self.Depths = np.insert(data['Depths'].values, 0,0)
            self.Depths_err = np.insert(data['Depths_err'].values, 0, 0.01)
        else:
            self.u_ages = U_series_ages
            self.low_age_err = U_series_ages_err_low / 2 # Data is read in at the 95% confidence
            self.high_age_err = U_series_ages_err_high / 2 # Data is read in at the 95% confidence
            self.Depths = data['Depths'].values
            self.Depths_err = data['Depths_err'].values
            
        
        # Chain stuff
        self.keys = ['Ages', 'Depths']
        
        self._move_funcs = [('Ages', self.Ages_move),
                            ('Depths', self.Depths_move)]


    def Log_Priors(self, theta):
        Age_Model, Depth_Model = theta

        diff_Model = np.diff(Age_Model)
        if np.any(diff_Model) < 0:
            return - np.inf

        lp = 0
    
        lp += np.sum(uniform.logpdf(Age_Model,
                                    loc = 0,
                                    scale = 1e6))
    
        lp += np.sum(uniform.logpdf(Depth_Model,
                                    loc = 0,
                                    scale = self.Depths.max()))
    
        return lp

    def Initial_Guesses_for_Model(self):
        initial_thetas = []

        for i in range(self.n_chains):
            log_prior = -np.inf
            while log_prior == -np.inf:
                
                delta_min = abs(0 - self.Depths[0])
                min_depths = np.random.uniform(self.Depths[0] - 2*self.Depths_err[0],
                                self.Depths[0] + 2*self.Depths_err[0])
                                
                max_depths = np.random.normal(self.Depths[-1] - 2*self.Depths_err[-1],
                                self.Depths[-1] + 2*self.Depths_err[-1])

                Depth_Model = np.linspace(min_depths, max_depths, 100)
                Model_Ages =  np.linspace(self.u_ages.min() - np.random.normal(0, 10),
                                              self.u_ages.max() + np.random.normal(0, 10), len(Depth_Model))
    
                    
                theta_initial =  Model_Ages, Depth_Model
    
                log_prior = self.Log_Priors(theta_initial)
    
                if log_prior != -np.inf:
                    initial_thetas.append(theta_initial)

        return initial_thetas

    def Logpdf_quick(self, mu, sigma, model):
        delta = mu - model
        ll =  -0.5 * np.square(delta / sigma) - np.log(sigma) - 0.5 * np.log(2*np.pi)
        return np.sum(ll)

    def Logpdf_asymm(self, mu, sigma_low, sigma_high, model):
        delta = mu - model
        sigma = np.where(delta < 0, sigma_low, sigma_high)
        ll = -0.5 * np.square(delta / sigma) - np.log(sigma) - 0.5 * np.log(2*np.pi)

        return np.sum(ll)


    def assign_unique_indices(self, Depths_Model, Depth_Observed):
        sorted_model_indices = np.argsort(Depths_Model)
        model_index_available = {idx: True for idx in sorted_model_indices}
    
        observed_indices = np.full(len(Depth_Observed), -1, dtype=int)
        sorted_obs_indices = np.argsort(Depth_Observed)
    
        for obs_idx in sorted_obs_indices:
            obs_depth = Depth_Observed[obs_idx]
            distances = np.abs(Depths_Model - obs_depth)
            for model_idx in np.argsort(distances):
                if model_index_available[model_idx]:
                    observed_indices[obs_idx] = model_idx
                    model_index_available[model_idx] = False
                    break
    
        return observed_indices


    def Log_Likelihood(self, theta):
        Age_Model, Depth_Model = theta
    
        unique_observed_indices = self.assign_unique_indices(Depth_Model, self.Depths)
        
        Depths_Model_observation = Depth_Model[unique_observed_indices]
        Age_Model_observation = Age_Model[unique_observed_indices]
        
    
        L1 = self.Logpdf_quick(self.Depths, self.Depths_err, Depths_Model_observation)
    
        L2 = self.Logpdf_asymm(self.u_ages, self.low_age_err, self.high_age_err, Age_Model_observation)
    
        return L1 + L2

    # Posterior
    def Log_Posterior(self, theta):
        """
        Function for determining Posterior
        ----------------------------------
        - This is Bayes-Price-Laplace Theorem
        in proportional and log form
        """
        lp = self.Log_Priors(theta)
        if not np.isfinite(lp):
            return - np.inf
        else:
            return lp + self.Log_Likelihood(theta)

    def Ages_move(self, theta, tuning_factor, index):
        """
        Modifies the Age_Model considering constraints based on the nearest observed heights (depths)
        and the corresponding observed ages plus/minus 2σ.
        """
        Age_Model, Depth_Model = theta
    
    
        Age_Model_prime = np.copy(Age_Model)
        Depth_Model_prime= np.copy(Depth_Model)
    
        Nmodel = len(Age_Model)
        # Calculate the current posterior probability
        log_posterior_current = self.Log_Posterior(theta)
    
        depth_indices = self.assign_unique_indices(Depth_Model, self.Depths)
    

        if index in depth_indices:
            sample_idx = np.where(depth_indices == index)[0][0]
            observed_age = self.u_ages[sample_idx]
            model_age = Age_Model[index]
            diff = observed_age - model_age
            step = np.random.normal(0, tuning_factor)
            if diff > 0:
                r = abs(step)
            else:
                r = -1 * abs(step)
        else:
            r = np.random.normal(0, tuning_factor)
                
            Age_Model_prime[index] += r

        # ===================
        # Age Conflict resolution
        # Keep things moving
        # ===================
        pivot = Age_Model_prime[index]

        # For all i > index: force Age_Model_prime[i] ≥ pivot
        Age_Model_prime[index+1:] = np.maximum(Age_Model_prime[index+1:], pivot)

        # For all i < index: force Age_Model_prime[i] ≤ pivot
        Age_Model_prime[:index] = np.minimum(Age_Model_prime[:index], pivot)
    
        # Update theta and calculate the new posterior
        theta_prime = Age_Model_prime, Depth_Model
        log_posterior_proposed = self.Log_Posterior(theta_prime)
        u = np.random.rand()
        # Make the Metropolis-Hastings decision
        if log_posterior_proposed > log_posterior_current or u <  np.exp(log_posterior_proposed - log_posterior_current):
            return Age_Model_prime, True  # Accept the new model
        else:
            return Age_Model, False  # Reject the new model


    def Depths_move(self, theta, tuning, index):
        """
        Modifies the Age_Model considering constraints based on the nearest observed heights (depths)
        and the corresponding observed ages plus/minus 2σ.
        """
        Age_Model, Depth_Model = theta
        # Calculate the current posterior probability
        log_posterior_current = self.Log_Posterior(theta)
    
        Depth_Model_prime = Depth_Model.copy()
        
        Depth_Model_prime[index] = Depth_Model[index] + np.random.normal(0, tuning)
                            
        # ===================
        # Conflict resolve depts
        # Just going to use Maximum accumulate here
        # ===================
        Depth_Model_prime = np.maximum.accumulate(Depth_Model_prime)
    
        # Update theta and calculate the new posterior
        theta_prime = Age_Model, Depth_Model_prime
        
        log_posterior_proposed = self.Log_Posterior(theta_prime)
        u = np.random.rand()
        
        # Make the Metropolis-Hastings decision
        if log_posterior_proposed > log_posterior_current or u <  np.exp(log_posterior_proposed - log_posterior_current):
            return Depth_Model_prime, True  # Accept the new model
        else:
            return Depth_Model, False  # Reject the new model


    def MCMC(self, theta, niters, chain_id):
        Age_Model, Depth_Model = theta
    
        if len(Age_Model) == 0 or len(Depth_Model) == 0:
            raise ValueError("Initial models cannot be empty")
            
        Nmodel = len(Age_Model)
        target_accept_rate = 0.234
        # Tuning Factors File Path
        tuning_factors_file = f'{self.sample_name}_tuning_factor_Age_Depth_{chain_id}.pkl'
        if os.path.exists(tuning_factors_file) and self.Start_from_pickles:
            with open(tuning_factors_file, 'rb') as f:
                tuning_factors = pickle.load(f)
        else:
            # If no file exists then initialize the tuning factors
            tuning_factors = {}
    
        for i in range(Nmodel):
            tuning_factors[f'Depths_{i}'] = np.mean(self.Depths_err)
            tuning_factors[f'Ages_{i}'] = np.mean(self.low_age_err)

        save_iters = int(self.iterations - self.burn_in)
    
        Age_Model_store = np.zeros((self.iterations, Nmodel))
        Depth_Model_store = np.zeros((self.iterations, Nmodel))
        Growth_rate_store = np.zeros((self.iterations, Nmodel - 1))
        posterior_store = np.zeros(self.iterations)
        
        Age_Model_store[0] = Age_Model
        Depth_Model_store[0] = Depth_Model
        Growth_rate_store[0] = np.diff(Age_Model)/np.diff(Depth_Model)
        posterior_store[0] = self.Log_Posterior(theta)
    
        proposal_counts = {p: 0 for p in tuning_factors}
        accept_counts   = {p: 0 for p in tuning_factors}
        ema_accept_rate = {p: 0.0 for p in tuning_factors}

              
        for i in range(1, niters):
            R = np.random.rand()
            if R < 0.85:
                move_name = 'Ages'
                Age_Model, Depth_Model = theta
                index = np.random.randint(0, Nmodel)
            
                new_values, accepted = self.Ages_move(theta,
                                    tuning_factors[f'{move_name}_{index}'],
                                    index)
            
                key = f'{move_name}_{index}'
                proposal_counts[key] += 1
                if accepted:
                    accept_counts[key] += 1
                    new_theta = new_values, Depth_Model
                    
                    theta = new_theta
                    
            if R >= 0.85:
                move_name = 'Depths'
                Age_Model, Depth_Model = theta
                index = np.random.randint(0, Nmodel)
            
                new_values, accepted = self.Depths_move(theta,
                                    tuning_factors[f'{move_name}_{index}'],
                                    index)
            
                key = f'{move_name}_{index}'
                proposal_counts[key] += 1
                if accepted:
                    accept_counts[key] += 1
                    new_theta = Age_Model, new_values
                    
                    theta = new_theta
    
            # Periodically adapt tuning factors
            # ---------------------------------------------------------
            #  Periodic adaptation of the tuning factors
            # ---------------------------------------------------------
            if i > 50 and i % 2000 == 0:
                for param in tuning_factors.keys():
                    if proposal_counts[param] == 0:
                        continue
                    a = accept_counts[param]
                    p = proposal_counts[param]
                        
                    block_accept_rate = accept_counts[param] / proposal_counts[param]
                    if block_accept_rate < target_accept_rate:
                        tuning_factors[param] *= 0.9
                    else:
                        tuning_factors[param] *= 1.1
                    accept_counts[param] = 0
                    proposal_counts[param] = 0
                    
            Age_Model_store[i,:] = theta[0]
            Depth_Model_store[i,:] = theta[1]
            Growth_rate_store[i,:] = np.diff(theta[0])/np.diff(theta[1])
            posterior_store[i] = self.Log_Posterior(theta)
    
    
            if (i + 1) % 5000 == 0:
                with open(f'{self.sample_name}_Age_Depth_theta_{chain_id}.pkl', 'wb') as f:
                    pickle.dump(theta, f)
                with open(f'{self.sample_name}_tuning_factor_Age_Depth_{chain_id}.pkl', 'wb') as f:
                    pickle.dump(tuning_factors, f)
                    
        return Age_Model_store, Depth_Model_store, Growth_rate_store, posterior_store



    """
    Check if files exist use pickles to start chains, else
    start the chain from random iniitalization
    """
    def check_starting_parameters(self):
        thetas = []
        for chain_id in range(self.n_chains):
            theta_pickle_file = f'{self.sample_name}_Age_Depth_theta_{chain_id}.pkl'
            if os.path.exists(theta_pickle_file) and self.Start_from_pickles:
                print(f'Pickles file exists and Start_from_pickles = {self.Start_from_pickles}')
                with open(theta_pickle_file, 'rb') as f:
                    theta_p = pickle.load(f)
                thetas.append(theta_p)
            else:
                # If no initial file exists, use initial guesses
                if not thetas or self.Start_from_pickles:  # Ensure it's only done once if needed
                    thetas = self.Initial_Guesses_for_Model()
                    print('Starting from random guesses')
                    
        return thetas
                        
    def Run_MCMC_Strat(self):
        iterations = self.iterations
        chain_ids = range(self.n_chains)
        all_thetas = self.check_starting_parameters()
        

        def run_chain(theta, chain_id):
            return self.MCMC(theta, self.iterations, chain_id)

        results = Parallel(n_jobs = 1)(delayed(run_chain)(theta,
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
        N_outputs = 4

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
            log_p.append(chain_dict[f"z4_{i}"])

        fig, ax = plt.subplots(1,1, figsize = (5,5))

        for i in range(self.n_chains):
            ax.plot(log_p[i],
               label = f'Chain {i + 1}')
    
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Posterior')
        ax.set_xscale('log')
        ax.legend(frameon = True, loc = 4, fontsize = 10, ncol = 2)

    def Get_Age_Model(self):
        result_dicts = self.Get_Results_Dictionary()
        Age_m = []
        for i in range(1, self.n_chains + 1):
            chain_dict = result_dicts[i-1]
            Age_m.append(chain_dict[f"z1_{i}"])
    
        c_age_m = np.concatenate(Age_m)  # shape (n_samples, n_points)
    
        low_age = np.percentile(c_age_m, 2.5, axis=0)
        high_age = np.percentile(c_age_m, 97.5, axis=0)
        age_median = np.percentile(c_age_m, 50, axis=0)
    
        return low_age, high_age, age_median

    def Get_Depth_Model(self):
        result_dicts = self.Get_Results_Dictionary()
        Depth_m = []
        for i in range(1, self.n_chains + 1):
            chain_dict = result_dicts[i-1]  # get the dictionary for chain i
            Depth_m.append(chain_dict[f"z2_{i}"])

        c_Depth_m = np.concatenate(Depth_m)
        
        low_depth, high_depth = np.percentile(c_Depth_m, [2.5, 97.5], axis = 0)
        depth_median = np.percentile(c_Depth_m, 50, axis = 0)

        return low_depth, high_depth, depth_median

    def Get_Age_Depth_Plot(self):
        err_asymm = [self.low_age_err * 2,
                    self.high_age_err * 2]

        low_age, high_age, age_median = self.Get_Age_Model()
        low_depth, high_depth, depth_median = self.Get_Depth_Model()
        
        fig, ax = plt.subplots(1, 1, figsize = (5,7))
        ax.errorbar(y = self.Depths,
                    x = self.u_ages,
                    xerr = err_asymm,
        fmt = 'o', markerfacecolor = 'dodgerblue',
        markeredgecolor = 'k', ecolor = 'k',
        capsize = 5, label = 'Bayesian\nAge\nEstimate (2$\sigma$)')

        ax.fill_betweenx(depth_median, low_age, high_age,
                         alpha = 0.3, color='navy',
                        label = '95% CI')

        ax.plot(high_age, depth_median,
                         ls = '--', color = 'navy', alpha = 0.4)
        
        ax.plot(low_age, depth_median,
                         ls = '--', color = 'navy', alpha = 0.4)
        
        ax.plot(age_median, depth_median,
                 lw = 1.5, color = 'royalblue', zorder = 2,
                label = 'Median Model')
                
        
        plt.ylim(self.Depths.max() + 2, self.Depths.min() - 2)
        plt.xlabel('Age (a)')
        plt.ylabel('Depth (mm)')
        ax.legend()


    def Save_Age_Depth_Model(self):
        low_age, high_age, age_median = self.Get_Age_Model()
        low_depth, high_depth, depth_median = self.Get_Depth_Model()

        df_age_depth = pd.DataFrame({"Depth_Median" : depth_median,
                                    "Depth_low_95" : low_depth,
                                    "Depth_high_95" : high_depth,
                                    "Age_Median" : age_median,
                                    "Age_low_95" : low_age,
                                    "Age_high_95" : high_age})

        df_age_depth.to_excel(f'{self.sample_name}_Age_Depth_Model.xlsx')







