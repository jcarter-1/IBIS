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
                              batch_size=1000,
                              max_attempts=5000000,
                              tolerance=1.00,
                              attempts_before_relax=2,
                              max_negative_ages=0,
                              desired_samples=10000,
                              printing = False):
        """
        Accumulates valid candidate initial 230Th values (and their uncertainties) that yield
        ages in acceptable stratigraphic order. Stops when desired_samples are collected or
        max_attempts reached.
        Returns:
            accepted_r02: np.array of initial Th values
            accepted_r02_err: np.array of uncertainties
            likelihood_weights: np.array of weights
        """
        accepted_r02 = []
        accepted_r02_err = []
        accepted_scores = []
        attempts_since_last_valid = 0
        total_attempts = 0
        current_score = -1e-9

        while total_attempts < max_attempts:
            # propose in log-space
            log_samps = np.random.uniform(np.log(0.1), np.log(200), batch_size)
            candidates = np.exp(log_samps)
            candidate_errs = candidates * np.random.uniform(0.001, 0.5, batch_size)

            # forward-model
            ages_all = np.zeros((batch_size, self.N_ratios))
            errs_all = np.zeros_like(ages_all)
            mask = np.ones(batch_size, bool)
            for i in range(self.N_ratios):
                for j in range(batch_size):
                    if not mask[j]: continue
                    try:
                        U = U_Series_Age_Equation(
                            self.r08[i], self.r08_err[i],
                            self.r28[i], self.r28_err[i],
                            self.r48[i], self.r48_err[i],
                            candidates[j], candidate_errs[j],
                            0,0,0,0)
                        age, err = U.Ages_And_Age_Uncertainty_Calculation_w_InitialTh()
                        ages_all[j,i] = age
                        errs_all[j,i] = err * 2
                    except:
                        mask[j] = False

            # validity filters
            mask &= ~np.isnan(ages_all).any(axis=1)
            mask &= (ages_all < 0).sum(axis=1) <= max_negative_ages
            mask &= ~((ages_all + 2*errs_all) < 0).any(axis=1)

            batch_idxs = []
            batch_scores = []
            for j in np.where(mask)[0]:
                # no local monotonic helper; use full stratigraphic score directly
                score = self.strat_log_likelihood(ages_all[j], errs_all[j], zeta=3)
                if score > current_score:
                    batch_idxs.append(j)
                    batch_scores.append(score)

            total_attempts += batch_size
            if batch_idxs:
                attempts_since_last_valid = 0
                accepted_r02.extend(candidates[batch_idxs])
                accepted_r02_err.extend(candidate_errs[batch_idxs])
                accepted_scores.extend(batch_scores)
                if printing:
                    print(f"Attempt {total_attempts}: +{len(batch_idxs)} accepted, total={len(accepted_r02)}")
            else:
                attempts_since_last_valid += batch_size

            if attempts_since_last_valid >= attempts_before_relax:
                current_score = current_score*10 if current_score > -1 else current_score - 1
                if printing:
                    print(f"Lower threshold: {current_score:.3e}")
                attempts_since_last_valid = 0

            if len(accepted_r02) >= desired_samples:
                print("Desired samples reached.")
                break

        if not accepted_r02:
            return np.array([]), np.array([]), np.array([])
        scored = np.array(accepted_scores)
        w = np.exp(scored - scored.max())
        w /= w.sum()
        return np.array(accepted_r02), np.array(accepted_r02_err), w

    def Get_Initial_Thoriums_For_Indices(self,
                                         ratio_indices,
                                         batch_size=1000,
                                         max_attempts=5000000,
                                         desired_samples=1000, printing = False):
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
            batch_size=batch_size,
            max_attempts=max_attempts,
            desired_samples=desired_samples,
            printing = printing)

        (self.N_ratios, self.r08, self.r08_err,
         self.r28, self.r28_err, self.r48, self.r48_err) = backup
        return vals, errs, w

    def Get_All_Windows(self,
                        window_size=3,
                        samples_per_window=1000,
                        batch_size=1000,
                        max_attempts=5000000,
                        printing = False):
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
            # truncate or pad to exactly samples_per_window
            count = min(len(v), samples_per_window)
            all_vals[i, :count] = v[:count]
            all_errs[i, :count] = e[:count]
            print(f"Window {i+1}/{n_win} (idx={idx}) → collected {count} samples")
        return all_vals, all_errs
        
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
        
        


    def get_kde_prior(self):
        # Retrieve initial samples, uncertainties, and scores.
        sample, sample_err = self.Get_All_Windows()
        sample = sample.flatten()
        sample_err = sample_err.flatten()
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
