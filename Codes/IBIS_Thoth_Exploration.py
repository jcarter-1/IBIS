import numpy as np
import pandas as pd
from scipy.optimize import fsolve, minimize
import pickle
from scipy.stats import norm, lognorm, truncnorm, gaussian_kde
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.interpolate import interp1d
import os

class U_Series_Age_Equation:
    def __init__(self, r08, r08_err,
                 r28, r28_err,
                 r48, r48_err,
                 r02_initial,
                 r02_initial_err,
                 r48_detrital,
                 r48_detrital_err,
                 r28_detrital,
                 r28_detrital_err,
                 rho_28_48 = 0.0,
                 rho_08_48 = 0.0,
                 rho_08_28 = 0.0,
                 r08_detrital = 0.0,
                 r08_detrital_err = 0.0):
        
        self.r08 = r08
        self.r08_err = r08_err
        self.r28 = r28
        self.r28_err = r28_err
        self.r48 = r48
        self.r48_err = r48_err
        self.r02_initial = r02_initial
        self.r02_initial_err = r02_initial_err
        self.r48_detrital = r48_detrital
        self.r48_detrital_err = r48_detrital_err
        self.r28_detrital = r28_detrital
        self.r28_detrital_err = r28_detrital_err
        self.lambda_230 = 9.1577e-6
        self.lambda_234 = 2.8263e-6
        self.lambda_230_err = 1.3914e-8
        self.lambda_234_err = 2.8234e-9
        self.rho_28_48 = rho_28_48
        self.rho_08_48 = rho_08_48
        self.rho_08_28 = rho_08_28
        self.r08_detrital = r08_detrital
        self.r08_detrital_err = r08_detrital_err
        


    def Age_Equation(self, T):
    
        A = self.r08
        B = 1 - np.exp(-self.lambda_230 * T)*(1 - self.r02_initial*self.r28)
        D = self.r48 - 1
        lam_diff =self.lambda_230 - self.lambda_234
        E = self.lambda_230 / lam_diff
        F = 1 - np.exp(-lam_diff*T)
        C = D * E * F
        return A - B - C



    def Age_solver(self, age_guess=1e4):
        """
        Solve the U-series age equation for a single age estimate.
        """
        func = lambda age: self.Age_Equation(age)
        t_solution = fsolve(func, age_guess)
        return t_solution[0]

    def Age_solver_Ludwig(self, age_guess=1e4):
        """
        Solve the U-series age equation for a single age estimate.
        """
        func = lambda age: self.Age_Equation_w_Ludwig(age)
        t_solution = fsolve(func, age_guess)
        return t_solution[0]


    def Ages_And_Age_Uncertainty_Calculation_w_InitialTh(self):
        """
        Age and uncertainty calculation
        decay constant uncertainties not included here
        """
        Age = self.Age_solver()
        
        # Compute lambda difference
        lam_diff = self.lambda_234 - self.lambda_230
        
        # Compute df/dT components
        df_dT_1 = self.lambda_230 * self.r28 * self.r02_initial * np.exp(-self.lambda_230 * Age)
        df_dT_2 = -self.lambda_230 * np.exp(-self.lambda_230 * Age)
        df_dT_3 = - (self.r48 - 1) * self.lambda_230 * np.exp(lam_diff * Age)
        df_dT = df_dT_1 + df_dT_2 + df_dT_3
        
        # Compute partial derivatives dt/dx_i
        dt_dr08 = -1 / df_dT
        dt_dr28 = (self.r02_initial * np.exp(-self.lambda_230 * Age)) / df_dT
        dt_dr02 = (self.r28 * np.exp(-self.lambda_230 * Age)) / df_dT
        dt_dr48 = - ((self.lambda_230 / lam_diff) * (1 - np.exp(lam_diff * Age))) / df_dT
    
        age_jacobian = np.array([dt_dr08,
                                 dt_dr28,
                                 dt_dr02,
                                 dt_dr48])
    
        cov_age = np.zeros((4,4))
        cov_age[0,0] = self.r08_err**2
        cov_age[0,1] = cov_age[1,0] = self.rho_08_28 * self.r08_err * self.r28_err
        cov_age[1,1] = self.r28_err**2
        cov_age[2,2] = self.r02_initial_err**2
        cov_age[3,3] = self.r48_err**2
        cov_age[1,3] = cov_age[3,1] = self.rho_28_48 * self.r48_err * self.r28_err
        cov_age[0,3] = cov_age[3,0] = self.rho_08_48 * self.r48_err * self.r08_err
    
        age_err = np.dot(age_jacobian, np.dot(cov_age, age_jacobian))
    
        return Age, np.sqrt(age_err)
        



class IBIS_Thoth_Robust:
    """
    Function to estimate a prior for the initial 230Th
    this will incorporate uncertainty into the measurement and provide a robust series of initial
    230Th estimations
    - We can then combined these into a complete weighted distribution (?)
    
    """
    def __init__(self, data, age_max, num_samples = 1000,
                 file_name='FILENAME', results_folder = None):

        self.data = data
        self.Th230_lam_Cheng =  9.1577e-06
        self.Th230_lam_Cheng_err = 1.3914e-08
        self.U234_lam_Cheng = 2.8263e-06
        self.U234_lam_Cheng_err =2.8234e-09
        self.file_name = file_name
        self.n_meas = data.shape[0]
        self.Thorium_prior = None
        self.Speleothem_params_filename = file_name +'_stal_parameters'
        self.N_ratios = data.shape[0]
        self.num_samples = num_samples
        self.age_max = age_max
        self.results_folder = results_folder
        
        # Unpack Ratios here
        self.r08 = self.data['Th230_238U_ratios'].values
        self.r28 = self.data['Th232_238U_ratios'].values
        self.r48 = self.data['U234_U238_ratios'].values
        self.r08_err= self.data['Th230_238U_ratios_err'].values
        self.r28_err = self.data['Th232_238U_ratios_err'].values
        self.r48_err = self.data['U234_U238_ratios_err'].values
        self.depths = self.data['Depths'].values
        self.num_samples = num_samples
        self.best = None
        self.best_err = None

        
  
    def Ages_And_Age_Uncertainty_Calculation_w_InitialTh(self):
        """
        Age and uncertainty calculation
        decay constant uncertainties not included here
        """
        Age = self.Age_solver()
        
        # Compute lambda difference
        lam_diff = self.lambda_234 - self.lambda_230
        
        # Compute df/dT components
        df_dT_1 = self.lambda_230 * self.r28 * self.r02_initial * np.exp(-self.lambda_230 * Age)
        df_dT_2 = -self.lambda_230 * np.exp(-self.lambda_230 * Age)
        df_dT_3 = - (self.r48 - 1) * self.lambda_230 * np.exp(lam_diff * Age)
        df_dT = df_dT_1 + df_dT_2 + df_dT_3
        
        # Compute partial derivatives dt/dx_i
        dt_dr08 = -1 / df_dT
        dt_dr28 = (self.r02_initial * np.exp(-self.lambda_230 * Age)) / df_dT
        dt_dr02 = (self.r28 * np.exp(-self.lambda_230 * Age)) / df_dT
        dt_dr48 = - ((self.lambda_230 / lam_diff) * (1 - np.exp(lam_diff * Age))) / df_dT
    
        age_jacobian = np.array([dt_dr08,
                                 dt_dr28,
                                 dt_dr02,
                                 dt_dr48])
    
        cov_age = np.zeros((4,4))
        cov_age[0,0] = self.r08_err**2
        cov_age[0,1] = cov_age[1,0] = self.rho_08_28 * self.r08_err * self.r28_err
        cov_age[1,1] = self.r28_err**2
        cov_age[2,2] = self.r02_initial_err**2
        cov_age[3,3] = self.r48**2
        cov_age[1,3] = cov_age[3,1] = self.rho_28_48 * self.r48_err * self.r28_err
        cov_age[0,3] = cov_age[3,0] = self.rho_08_48 * self.r48_err * self.r08_err
    
        age_err =  age_jacobian @ cov_age @ age_jacobian.T
    
        return Age, np.sqrt(age_err)
        
    def Age_solver(self, age_guess=1e4):
        """
        Solve the U-series age equation for a single age estimate.
        """
        func = lambda age: self.Age_Equation(age)
        t_solution = fsolve(func, age_guess)
        return t_solution[0]
        
    
    def compute_ages(self, init_thorium, init_thorium_err):
        """
        For a given candidate initial thorium value and its uncertainty,
        compute the suite of ages and age uncertainties.
        """
        ages = np.zeros(self.N_ratios)
        age_errs = np.zeros(self.N_ratios)
        for idx in range(self.N_ratios):
            try:
                U_age = U_Series_Age_Equation(
                    self.r08[idx], self.r08_err[idx],
                    self.r28[idx], self.r28_err[idx],
                    self.r48[idx], self.r48_err[idx],
                    init_thorium, init_thorium_err,
                    0, 0, 0, 0)
                age, age_err = U_age.Ages_And_Age_Uncertainty_Calculation_w_InitialTh()
                ages[idx] = age
                # Inflate the age uncertainty if desired
                age_errs[idx] = age_err
            except Exception as e:
                ages[idx] = np.nan
                age_errs[idx] = np.nan
        return ages, age_errs
        


    def _strat_fraction(self, ages, errs):
        """
        Fraction of adjacent pairs that satisfy:
          • same‐depth → |Δage| ≤ combined_unc
          • deeper sample older (within unc)
        """

        count = 0
        total = self.N_ratios - 1
        for i in range(total):
            σ = np.sqrt(errs[i]**2 + errs[i+1]**2)
            Δ = ages[i+1] - ages[i]

            if self.depths[i] == self.depths[i+1]:
                # same depth: require agreement within σ
                if abs(Δ) <= σ:
                    count += 1
            else:
                # stratigraphic: deeper (j) should be at least as old as i
                if Δ >= -σ:
                    count += 1

        return count / total


    def _strat_loglik(self,  ages, errs):
        """
        Sum log‐likelihood penalties over all i<j:
          • same‐depth → logpdf(Δage | 0, σ)
          • deeper sample older: if Δ<0, add logcdf(Δ/σ)
        """
        ll = 0.0
        N  = len(ages)
        # no need to sort here, we just compare every pair
        for i in range(N-1):
            for j in range(i+1, N):
                σ    = np.sqrt(errs[i]**2 + errs[j]**2)
                Δ    = ages[j] - ages[i]
                if self.depths[i] == self.depths[j]:
                    # enforce equality
                    ll += norm.logpdf(Δ, loc=0.0, scale=σ)
                else:
                    if Δ < 0:
                        ll += norm.logcdf(Δ / σ)
        return ll


    def _combined_score(self,ages, errs):
        """
        A single metric combining:
          • strat‐fraction
          • logistic of strat‐loglik
          • penalty for any negative ages
        """
        frac = self._strat_fraction(ages, errs)
        ll   = self._strat_loglik(ages, errs)
        logistic = 1.0 / (1.0 + np.exp(-ll))

        # penalize negative ages strongly
        mean_neg = np.mean(np.minimum(ages, 0.0))
        penalty  = -100.0 * abs(mean_neg)

        return frac + logistic + penalty
        
    def uniform_log_sample(self, lower=0.1, upper=500, n_samp = 1):
        """
        Then lets sample more gently here
        """
        

        u = np.random.uniform(0, 1, n_samp)
        candidate = lower * (upper/lower)**u
        return candidate
        
    def detrital_sample(self, n_samp):
        a, b = (0 - 0.8)/0.4, np.inf
        
        return truncnorm(a = a, b = b, loc = 0.8, scale = 0.4).rvs(size = n_samp)
        

    def all_thorium(self, n_samp):
        z = norm.ppf(0.99)
        C = np.log(0.8) - np.log(500)
        disc = z**2 - 4*C
        sigma = (-z + np.sqrt(disc)) / 2
        mu = np.log(0.8) + sigma**2
        
        return lognorm(s = sigma, scale = np.exp(mu)).rvs(size = n_samp)
        
    def boutique_thoriums(self, batch_size):
        # decide which draws are from the broad vs. tight component
        detrital = np.random.rand(batch_size) < 0.5
    
        # allocate array
        candidates = np.empty(batch_size)
    
        # Detrital_Vale
        n_detrital = detrital.sum()
        candidates[detrital] = self.detrital_sample(n_detrital)
    
        # tight‐cluster samples
        n_uniform = batch_size - n_detrital
        candidates[~detrital] = self.all_thorium(n_samp = n_uniform)
    
        # clip into [lower, upper]
        np.clip(candidates, 0.01, 50, out=candidates)
        return candidates

    def Get_Initial_Thoriums(self,
                              batch_size=1000,       # Number of candidates     per batch
                              max_attempts=500000,
                              attempts_before_relax = 100,
                              desired_samples = 1000):
        """
        Accumulates valid candidate initial 230Th values (and their     uncertainties) that yield
        ages in acceptable stratigraphic order. All valid candidates are    saved until a total
        of max_attempts have been attempted or until a specified number of  accepted candidates
        (e.g., 10,000) have been collected. Then, the function computes the     likelihood weights
        (from the likelihood scores) and returns the candidate values,  uncertainties, and
        likelihood weights.
    
        Returns:
            accepted_r02       : (np.array) Accepted candidate initial 230Th    values.
            accepted_r02_err   : (np.array) Corresponding candidate     uncertainties.
            likelihood_weights : (np.array) Likelihood weights computed from    the scores.
        """
        accepted_r02 = []
        accepted_r02_err = []
        accepted_scores = []
    
        attempts_since_last_valid = 0
        total_attempts = 0
        # Set an initial threshold for the likelihood score.
        current_score = 0 # Adjust as needed
    
        # Set the desired number of samples to stop early.
        desired_samples = self.num_samples
    
        while total_attempts < max_attempts:
            # Generate candidate initial thorium values.
            candidates = self.boutique_thoriums(batch_size)
            # Generate candidate uncertainties (as a fraction of the    candidate value).
            candidate_errs = (np.array([np.random.uniform(0.001, 0.5)
                                        for _ in range(batch_size)])
                              * candidates)
    
            # Prepare storage for ages and uncertainties for each candidate     and each ratio.
            ages_all = np.zeros((batch_size, self.N_ratios))
            age_errs_all = np.zeros((batch_size, self.N_ratios))
            valid_mask = np.ones(batch_size, dtype=bool)
    
            # Loop through each ratio and each candidate to compute ages.
            for idx in range(self.N_ratios):
                for j in range(batch_size):
                    try:
                        U_age = U_Series_Age_Equation(
                            self.r08[idx], self.r08_err[idx],
                            self.r28[idx], self.r28_err[idx],
                            self.r48[idx], self.r48_err[idx],
                            candidates[j], candidate_errs[j],
                            0, 0, 0, 0)
                        age, age_err =  U_age.Ages_And_Age_Uncertainty_Calculation_w_InitialTh()
                        ages_all[j, idx] = age
                        age_errs_all[j, idx] = age_err
                    except Exception as e:
                        ages_all[j, idx] = np.nan
                        age_errs_all[j, idx] = np.nan
                        valid_mask[j] = False
    
            # Filter out candidates with invalid results.
            valid_mask &= ~np.any(np.isnan(ages_all), axis=1)
            #valid_mask &= (np.sum(ages_all < 0, axis=1) <= max_negative_ages)
            valid_mask &= ~np.any(ages_all + 2 * age_errs_all < 0, axis=1)
            valid_mask &= ~np.any(ages_all - 2 * age_errs_all > self.age_max, axis=1)

            batch_valid_indices = []
            batch_scores = []  # Likelihood scores for candidates in this   batch
    
            # Evaluate each candidate that passed the validity check.
            for j in np.where(valid_mask)[0]:
                # Calculate the stratigraphic likelihood score for candidate    j.
                strat_score = self._combined_score(ages_all[j, :],age_errs_all[j, :])
                # Accept candidate if the likelihood score exceeds the current threshold.
                if strat_score > current_score:
                    batch_valid_indices.append(j)
                    batch_scores.append(strat_score)
    
            total_attempts += batch_size
    
            if len(batch_valid_indices) == 0:
                attempts_since_last_valid += batch_size
            else:
                attempts_since_last_valid = 0
                # Save the valid candidates from this batch.
                accepted_r02.extend(candidates[batch_valid_indices])
                accepted_r02_err.extend(candidate_errs[batch_valid_indices])
                accepted_scores.extend(batch_scores)
                print(f"Batch at attempt {total_attempts}: Found    {len(batch_valid_indices)} valid candidates (total saved:   {len(accepted_r02)}).")
    
            # If no valid candidate has been found for a while, lower the   acceptable likelihood threshold.
            if attempts_since_last_valid >= attempts_before_relax:
                if current_score == 0:
                    current_score += -0.5
                if current_score > -1:
                    current_score *= 10
                if current_score <= -1:
                    current_score -= 1
                if current_score <= -110:
                    current_score *= 1.1

                print(f"Lowering likelihood threshold to    {current_score:.10f}.")
                attempts_since_last_valid = 0
    
            # Check if we've collected enough samples.
            if len(accepted_r02) >= desired_samples:
                print(f"Desired sample size of {desired_samples} reached.")
                break
    
        # Convert lists to arrays.
        accepted_r02 = np.array(accepted_r02)
        accepted_r02_err = np.array(accepted_r02_err)
        accepted_scores = np.array(accepted_scores)
    
        if len(accepted_r02) == 0:
            print(f"Warning: No valid candidates found after {total_attempts} attempts.")
            return np.array([]), np.array([]), np.array([])
    
        # Compute likelihood weights in a numerically stable way.
        likelihood_weights = np.exp(accepted_scores - np.max(accepted_scores))
        likelihood_weights /= np.sum(likelihood_weights)
    
        # Return all accepted candidates, their uncertainties, and the  likelihood weights.
        return accepted_r02, accepted_r02_err, likelihood_weights
        
        
    """
    Now think about segments
    ------------------------
    """
    def Get_Initial_Thoriums_For_Indices(self,
                                         ratio_indices,
                                         batch_size=5000,
                                         max_attempts=5000000,
                                         desired_samples=100, printing = True):
        """
        Sample initial-Th using only a subset of ratios given by indices.
        """
        backup = (self.N_ratios, self.r08, self.r08_err,
                  self.r28, self.r28_err, self.r48, self.r48_err)
        idx = ratio_indices
        self.N_ratios = len(idx)
        self.r08, self.r08_err = self.r08[idx], self.r08_err[idx]
        self.r28, self.r28_err = self.r28[idx], self.r28_err[idx]
        self.r48, self.r48_err = self.r48[idx], self.r48_err[idx]

        vals, errs, w = self.Get_Initial_Thoriums(
                              batch_size=1000,       # Number of candidates     per batch
                              max_attempts=5000000,
                              attempts_before_relax = 100,
                              desired_samples = desired_samples)

        (self.N_ratios, self.r08, self.r08_err,
         self.r28, self.r28_err, self.r48, self.r48_err) = backup
        return vals, errs, w

    def Get_All_Windows(self,
                        window_size=3,
                        samples_per_window=100,
                        batch_size=1000,
                        max_attempts=5000000,
                        printing = True):
        """
        Slide a window of `window_size` across the N ratios, collect
        `samples_per_window` for each window.
        Returns arrays shape (n_windows, samples_per_window).
        """
        n = self.N_ratios
        n_win = n - window_size + 1
        all_vals = np.full((n_win, samples_per_window), np.nan)
        all_errs = np.full((n_win, samples_per_window), np.nan)
        for i in range(n_win):
            idx = list(range(i, i + window_size))
            v, e, _ = self.Get_Initial_Thoriums_For_Indices(
                ratio_indices=idx,
                batch_size=batch_size,
                max_attempts=max_attempts,
                desired_samples=samples_per_window,
                printing = printing)

            if len(v) == 0 and len(e) == 0:
                # nothing to do for this window → leave NaNs
                if printing:
                    print(f"Window {i+1}/{n_win} (idx={idx}) → no valid samples")
                continue
            #print(f"Warning: No valid candidates found after {total_attempts} attempts.")
            #return 0, 0, 0
            
            # truncate or pad to exactly samples_per_window
            count = min(len(v), samples_per_window)
            all_vals[i, :count] = v[:count]
            all_errs[i, :count] = e[:count]
            print(f"Window {i+1}/{n_win} (idx={idx}) → collected {count} samples")
        # now flatten and drop any NaNs (or Infs) before returning
        flat_vals = all_vals.ravel()
        flat_errs = all_errs.ravel()

        # build a mask of finite entries
        mask = np.isfinite(flat_vals)

        # filter
        flat_vals = flat_vals[mask]
        flat_errs = flat_errs[mask]

        return flat_vals, flat_errs
        
    def Segements(self):
        vals, errs = self.Get_All_Windows(samples_per_window= int(self.num_samples/100))
        
        r02 = vals.flatten()
        r02_err = errs.flatten()
    
        return r02, r02_err
        
    def Segments(self):
        """
        Returns:
          seg_vals: list of length n_windows, each an array of size samples_per_window
          seg_errs: list of length n_windows, each an array of size samples_per_window
        """
        # ask for the full 2D output from Get_All_Windows
        # we'll re‐implement its logic here so we keep the matrix shape
        window_size       = 3
        samples_per_window = int(self.num_samples/100)
        n                 = self.N_ratios
        n_win             = n - window_size + 1

        # hold the raw values
        seg_vals = np.full((n_win, samples_per_window), np.nan)
        seg_errs = np.full((n_win, samples_per_window), np.nan)

        for i in range(n_win):
            vals_i, errs_i, _ = self.Get_Initial_Thoriums_For_Indices(
                ratio_indices    = list(range(i, i+window_size)),
                desired_samples  = samples_per_window
            )
            count = min(len(vals_i), samples_per_window)
            seg_vals[i, :count] = vals_i[:count]
            seg_errs[i, :count] = errs_i[:count]

        # turn each row into its own array and return
        return [seg_vals[i, :][~np.isnan(seg_vals[i, :])]   for i in range(n_win)], \
               [seg_errs[i, :][~np.isnan(seg_errs[i, :])]   for i in range(n_win)]
        

    def get_kde_prior(self,
                    bw_method     = 'silverman',
                    include_global= True,
                    global_boost  = 50.0):
        """
        Builds one KDE per window + optional global KDE, then
        mixes them with equal window‐weights but a boosted global weight.
        
        global_boost = how many times more weight the GLOBAL KDE gets
                    compared to _one_ window KDE
        """
        kdes    = []
        weights = []
    
        # 1) Global expert
        if include_global:
            sample, _, _ = self.Get_Initial_Thoriums(self.num_samples)
            if sample.size > 0:
                kdes.append( gaussian_kde(np.log(sample), bw_method=bw_method) )
                weights.append(global_boost)
            else:
                print("no valid samples - windows only")
    
        # 2) Per‐window experts
        seg_vals_list, _ = self.Segments()  # now returns list of arrays
        for vals in seg_vals_list:
            vals = np.array(vals)
            if vals.size > 0:
                kdes.append(gaussian_kde(np.log(vals), bw_method=bw_method) )
                weights.append(1.0)
    
        # 3) normalize mixture weights
        weights = np.array(weights)
        weights = weights / weights.sum()
    
        # 4) build a common grid in log‐space
        all_draws = np.concatenate([sample] + seg_vals_list)
        lo  = np.min(all_draws)
        hi  = np.percentile(all_draws, 99.9)
        grid_log = np.linspace(np.log(lo), np.log(hi), 2000)
        grid     = np.exp(grid_log)
        dx       = np.diff(grid).mean()
    
        # 5) evaluate & combine
        pdf = np.zeros_like(grid)
        for w, kde in zip(weights, kdes):
            pdf += w * kde(grid_log)
        # Jacobian back‐transform
        pdf /= grid
        # final normalization
        pdf /= (pdf.sum() * dx)
    
        # 6) wrap in an interpolator
        self.Thorium_prior = interp1d(
            grid, pdf,
            kind='linear',
            bounds_error=False,
            fill_value=1e-12
        )
        return self.Thorium_prior
        
    def get_values_and_weights(self):
        kdes    = []
        weights = []
    
        # 1) Global expert
        if include_global:
            sample, _, _ = self.Get_Initial_Thoriums(self.num_samples)
            if sample.size > 0:
                kdes.append( gaussian_kde(np.log(sample), bw_method=bw_method) )
                weights.append(global_boost)
            else:
                print("no valid samples - windows only")
    
        # 2) Per‐window experts
        seg_vals_list, _ = self.Segments()  # now returns list of arrays
        for vals in seg_vals_list:
            vals = np.array(vals)
            if vals.size > 0:
                kdes.append(gaussian_kde(np.log(vals), bw_method=bw_method) )
                weights.append(1.0)
    
        # 3) normalize mixture weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return kdes, weights

    def save_thor_prior(self):
        # make sure the prior is computed
        if self.Thorium_prior is None:
            self.get_kde_prior()

        # build a filename with extension
        fname = f"{self.file_name}.pkl"
        path  = os.path.join(self.results_folder, fname)

        # write the pickle
        with open(path, 'wb') as f:
            pickle.dump(self.Thorium_prior, f)
        print(f"✅  Thorium prior saved to {path}")

        return path

    def load_thor_prior(self):
        # same filename we used for saving
        fname = f"{self.file_name}.pkl"
        path  = os.path.join(self.results_folder, fname)

        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.Thorium_prior = pickle.load(f)
            print(f"♻️  Thorium prior loaded from {path}")
        else:
            print(f"⚠️  No prior at {path}; computing & saving now.")
            self.save_thor_prior()
