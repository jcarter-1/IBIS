import numpy as np
import pandas as pd
from scipy.optimize import fsolve, minimize
import pickle
from scipy.stats import norm

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
    
        A = self.r08 - self.r28 * self.r02_initial * np.exp(-self.lambda_230 * T)
        B = 1 - np.exp(-self.lambda_230 * T)
        D = self.r48 - 1
        E = self.lambda_230 / (self.lambda_234 - self.lambda_230)
        F = 1 - np.exp((self.lambda_234 - self.lambda_230)*T)
        C = D * E * F
        return A - B + C



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
    def __init__(self, data, num_samples,
                 file_name='FILENAME',
                Th230_lam_Cheng = 9.1577e-06,
                Th230_lam_Cheng_err = 1.3914e-08,
                U234_lam_Cheng = 2.8263e-06,
                U234_lam_Cheng_err= 2.8234e-09):

        self.data = data
        self.Th230_lam_Cheng = Th230_lam_Cheng
        self.Th230_lam_Cheng_err = Th230_lam_Cheng_err
        self.U234_lam_Cheng = U234_lam_Cheng
        self.U234_lam_Cheng_err = U234_lam_Cheng_err
        self.file_name = file_name
        self.n_meas = data.shape[0]
        self.Thorium_prior = None
        self.Speleothem_params_filename = file_name +'_stal_parameters'
        self.N_ratios = data.shape[0]
        
        
        # Unpack Ratios here
        self.r08 = self.data['Th230_238U_ratios'].values
        self.r28 = self.data['Th232_238U_ratios'].values
        self.r48 = self.data['U234_U238_ratios'].values
        self.r08_err= self.data['Th230_238U_ratios_err'].values
        self.r28_err = self.data['Th232_238U_ratios_err'].values
        self.r48_err = self.data['U234_U238_ratios_err'].values
        self.num_samples = num_samples

    def check_frac_in_strat_order(self, ages, age_errs):
        """
        Say 80% or above is okay and see what happens here
        """
        diff_ages = np.diff(ages)
        err_diff = np.sqrt(age_errs[:-1]**2 + age_errs[1:]**2)
        
        n_intervals = len(diff_ages)
        n_in_order = 0
        
        for i in range(n_intervals):
            
            if not (diff_ages[i] < 0 and abs(diff_ages[i]) > err_diff[i]):
                n_in_order += 1

        fraction_in_order = n_in_order / n_intervals
        return fraction_in_order
        
        
  
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
        
        
    def uniform_log_sample(self, lower=0.01, upper=300):
        # Sample uniformly in log space:
        u = np.random.uniform(0, 1)
        candidate = lower * (upper/lower)**u
        return candidate

        
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
        
    def strat_log_likelihood(self, ages, age_errs, zeta=3):
        """
        Computes the stratigraphic log likelihood based on the ordering of ages.
        """
        LL = 0.0
        n = len(ages)
        for i in range(n):
            for j in range(i + 1, n):
                diff = ages[j] - ages[i]
                combined_unc = np.sqrt(age_errs[i]**2 + age_errs[j]**2)
                if diff > 0:
                    LL += 0  # no penalty if the later age is greater than the earlier one
                elif abs(diff) > combined_unc:
                    # Use the combined uncertainty directly
                    LL += norm.logcdf(diff / combined_unc)
                else:
                    # Increase the uncertainty by a factor zeta when ages are very close
                    LL += norm.logcdf(diff / (zeta * combined_unc))
        return LL

    def Get_Initial_Thoriums(self,
                              batch_size=1000,       # Number of candidates     per batch
                              max_attempts=5000000,    # Total number of   candidate attempts
                              tolerance=1.00,         # Not adapted in this     version; kept constant
                              attempts_before_relax=2,  # Attempts before  lowering the threshold
                              max_negative_ages=0):
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
        # A helper function to compute a single age using a given ratio index.
        def compute_age_for_ratio(candidate, candidate_err, ratio_index):
            # Replace with the appropriate call to U_Series_Age_Equation etc.
            U_age = U_Series_Age_Equation(
                        self.r08[ratio_index], self.r08_err[ratio_index],
                        self.r28[ratio_index], self.r28_err[ratio_index],
                        self.r48[ratio_index], self.r48_err[ratio_index],
                        candidate, candidate_err,
                        0, 0, 0, 0)
            age, age_err = U_age.Ages_And_Age_Uncertainty_Calculation_w_InitialTh()
            # Optionally multiply age_err by some factor (as in your original code)
            return age, age_err * 2
         # Step 1: Get candidate samples that produce a valid first age.
        ccepted_candidates = []  # list of tuples: (candidate, candidate_err, [age0, ...], [age_err0, ...])
        attempts = 0
        while len(accepted_candidates) < samples_per_age and attempts < max_attempts:
           candidate = self.uniform_log_sample(lower=0.1, upper=200)
           candidate_err = np.random.uniform(0.001, 0.5) * candidate
        
           try:
               # Compute the first age (ratio index 0)
               age0, age_err0 = compute_age_for_ratio(candidate, candidate_err, ratio_index=0)
               # Apply your acceptance criteria for the first age.
               # For example, you could require age0 > 0:
               if age0 > 0:
                   accepted_candidates.append((candidate, candidate_err, [age0], [age_err0]))
           except Exception as e:
               # If computation fails, skip this candidate.
               pass
           attempts += 1
        
        if len(accepted_candidates) == 0:
           print(f"Warning: No valid candidates found for the first age after {attempts} attempts.")
           return np.array([]), np.array([]), np.array([])

        for ratio_index in range(1, self.N_ratios):
           new_accepted = []
           # First, try to "extend" existing candidates.
           for candidate, candidate_err, ages_so_far, age_errs_so_far in accepted_candidates:
               try:
                   age_k, age_err_k = compute_age_for_ratio(candidate, candidate_err, ratio_index=ratio_index)
                   # For stratigraphic order, enforce that the new age is greater than the last accepted age.
                   if age_k > ages_so_far[-1]:
                       new_accepted.append(
                           (candidate, candidate_err, ages_so_far + [age_k], age_errs_so_far + [age_err_k])
                       )
               except Exception:
                   continue
        
           # If new candidates are insufficient in number, perform additional sampling for this ratio.
           extended_attempts = 0
           while len(new_accepted) < samples_per_age and extended_attempts < max_attempts:
               candidate = self.uniform_log_sample(lower=0.1, upper=200)
               candidate_err = np.random.uniform(0.001, 0.5) * candidate
               try:
                   # Compute the full set of ages for the candidate up to ratio_index.
                   temp_ages = []
                   temp_age_errs = []
                   valid_candidate = True
                   for idx in range(ratio_index + 1):
                       age, age_err = compute_age_for_ratio(candidate, candidate_err, ratio_index=idx)
                       # Ensure that ages are in ascending order.
                       if idx > 0 and age <= temp_ages[idx - 1]:
                           valid_candidate = False
                           break
                       temp_ages.append(age)
                       temp_age_errs.append(age_err)
                   if valid_candidate:
                       new_accepted.append((candidate, candidate_err, temp_ages, temp_age_errs))
               except Exception:
                   pass
               extended_attempts += 1
        
           if len(new_accepted) == 0:
               print(f"Warning: No candidates passed the stratigraphic test at ratio index {ratio_index}.")
               return np.array([]), np.array([]), np.array([])
           # For the next round, use the newly accepted candidates.
           accepted_candidates = new_accepted
           print(f"After processing ratio index {ratio_index}, {len(accepted_candidates)} candidates remain.")
    
           # At this point, accepted_candidates contains candidate tuples that have valid ages for all ratios.
           final_candidates = np.array([cand for cand, _, _, _ in accepted_candidates])
           final_candidates_err = np.array([err for _, err, _, _ in accepted_candidates])
        
        # Optionally, if you have likelihood scores from each stage or from a combined stratigraphic score,
        # compute weights in a numerically stable way. For example:
        # (In this pseudocode we have not computed explicit scores, so you might compute them as needed.)
        # likelihood_scores = np.array([...])
        # likelihood_weights = np.exp(likelihood_scores - np.max(likelihood_scores))
        # likelihood_weights /= np.sum(likelihood_weights)
        
        # For demonstration, we return equal weights.
        likelihood_weights = np.ones(len(final_candidates)) / len(final_candidates)
        
        return final_candidates, final_candidates_err, likelihood_weights
            
                
    def save_thor_prior(self):
        if self.Thorium_prior is None:
            self.get_kde_prior()

        with open(self.file_name, 'wb') as f:
            pickle.dump(self.Thorium_prior, f)
            print(f'Ages, Unceratinties, and Maximum age saved to {self.file_name}')
            
        return self.Thorium_prior

    def load_thor_prior(self):
        if os.path.exists(self.file_name):
            with open(self.file_name, 'rb') as f:
                self.Thorium_prior = pickle.load(f)
                print('Thorium Prior Loaded from file')
        else:
            print('Thorium Prior does not exist, generating then saving to filepath')
            self.save_thor_prior()
        
        
    def get_kde_prior2(self):
        # Retrieve initial samples, uncertainties, and scores.
        sample, sample_err, scores = self.Get_Initial_Thoriums()
         
        # Generate bootstrapped samples incorporating uncertainties.
        boot_samples = np.random.normal(loc=sample, scale=sample_err, size=(10000, len(sample)))
        # Flatten to 1D array.
        boot_samples = boot_samples.flatten()
        
        # Combine scores with an inverse uncertainty term to prefer samples with lower uncertainties.
        epsilon = 1e-6  # Small constant to avoid division by zero
        # Repeat the scores and uncertainties for each bootstrap replicate.
        repeated_scores = np.tile(scores, 10000)
        repeated_uncertainties = np.tile(sample_err, 10000)
        # Weight is proportional to the score and inversely proportional to the uncertainty.
        weights = repeated_scores / (repeated_uncertainties + epsilon)
        
        # Filter out non-physical (e.g., negative) samples.
        valid_mask = boot_samples > 0
        valid_samples = boot_samples[valid_mask]
        valid_weights = weights[valid_mask]
        # Normalize weights so they sum to 1.
        valid_weights = valid_weights / np.sum(valid_weights)
        
        # Compute the Gaussian KDE with the weighted samples.
        from scipy.stats import gaussian_kde
        kde_samples = gaussian_kde(valid_samples, weights=valid_weights, bw_method='silverman')
        higher = np.percentile(valid_samples, 99.99)
        # Evaluate the KDE on a grid.
        grid_eval = np.linspace(0, higher, 500)
        density_vals = kde_samples(grid_eval)
        
        # Normalize the density values so that the area under the curve is 1.
        dx = grid_eval[1] - grid_eval[0]
        density_vals = density_vals / (np.sum(density_vals) * dx)
        
        # Create an interpolation function for the density (your prior).
        from scipy.interpolate import interp1d
        kde_interpolate = interp1d(grid_eval, density_vals, kind='linear',
                                   bounds_error=False, fill_value=1e-12)
        self.Thorium_prior = kde_interpolate
        return self.Thorium_prior


    def get_kde_prior(self):
        # Retrieve initial samples, uncertainties, and scores.
        sample, sample_err, scores = self.Get_Initial_Thoriums()
            
        # Generate a 2D array where each row is a resample.
        # The size is (5000, number of samples in 'sample')
        resamples = np.random.normal(loc=sample, scale=sample_err, size=(5000, np.size(sample)))

        # Flatten the array into 1D.
        valid_samples = resamples.flatten()

        # Filter to keep only positive samples.
        valid_samples = valid_samples[valid_samples > 0]
                
        # Compute the Gaussian KDE with the weighted samples.
        from scipy.stats import gaussian_kde
        kde_samples = gaussian_kde(valid_samples, bw_method='silverman')
        higher = np.percentile(valid_samples, 99.99)
        # Evaluate the KDE on a grid.
        grid_eval = np.linspace(0, higher, 500)
        density_vals = kde_samples(grid_eval)
        
        # Normalize the density values so that the area under the curve is 1.
        dx = grid_eval[1] - grid_eval[0]
        density_vals = density_vals / (np.sum(density_vals) * dx)
        
        # Create an interpolation function for the density (your prior).
        from scipy.interpolate import interp1d
        kde_interpolate = interp1d(grid_eval, density_vals, kind='linear',
                                   bounds_error=False, fill_value=1e-12)
        self.Thorium_prior = kde_interpolate
        return self.Thorium_prior
