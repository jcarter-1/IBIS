from ibis_codes_KDE_Cov import IBIS_Configuration
from ibis_codes_KDE_Cov import IBIS_Bounds_and_Uncertainties
from ibis_codes_KDE_Cov import IBIS_Thoth_Exploration
from ibis_codes_KDE_Cov import IBIS_MCMC_Initial_Th
from ibis_codes_KDE_Cov import IBIS_strat
from ibis_codes_KDE_Cov import USeries_Age_Equations
from ibis_codes_KDE_Cov import IBIS_strat_Hiatus
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde, lognorm
from datetime import datetime


class IBIS:
    def __init__(self, filepath, sample_name, MCMC_samples = 5000, MCMC_burn_in = 1000, MCMC_Strat_samples = 50000,
                n_chains = 3, Start_from_pickles = True, Top_Age_Stal = False, Hiatus = False, show_bird = True):
        self.filepath = filepath
        # Configure the Data
        self.config = IBIS_Configuration.IBIS_Configuration_Class(self.filepath)
        # Get refined dataset
        self.df_reduced = self.config.Get_Measured_Ratios()
        # Sample
        self.sample_name = sample_name
        # Kdes_filepath
        self.kdes_name = self.sample_name + '_prior'
        # Bounds_Filename
        self.bounds_file_name = self.sample_name + '_bounds'
        # Number of samples
        self.MCMC_samples = MCMC_samples
        self.MCMC_burn_in = MCMC_burn_in
        # Number of Strat samples
        self.MCMC_Strat_samples = MCMC_Strat_samples
        # Number of chains
        self.n_chains = n_chains
        # Check Include Detrital 
        self.Include_Detrital = None
        # Check
        self.are_there_bounds = False
        self.Thorium_prior_exist = False
        # Check
        self.are_there_speleothem_parameters = False
        # Check
        self.set_up_the_chain = False
        # Check
        self.Chain_run = False
        self.Chain_strat_run = False
        # Initialize 
        self.Start_from_pickles = Start_from_pickles
        # Create a results folder
        self.output_dir = os.path.abspath(f'{self.sample_name}_results_folder')
        os.makedirs(self.output_dir, exist_ok = True)

        # Check for Hiatus and Top Stal Collection Age
        self.Top_Age_Stal = Top_Age_Stal
        if self.Top_Age_Stal: 
            self.collect_data = input('Please input date of sample collection in dd-mm-yyyy format: ')
            self.Top_Age_Stal  = self.Get_Top_Age()
        self.Hiatus_Check = Hiatus

        if self.Hiatus_Check:
            while True:
                try:
                    self.Hiatus_start = float(input(
                        "Please input start (highest point) of the observed Hiatus "
                        "– match units of the input data: "
                    ))
                    break
                except ValueError:
                    print("⚠️  Invalid number. Please enter a decimal or integer value.")
        
            while True:
                try:
                    self.Hiatus_end = float(input(
                        "Please input end (lowest point) of the observed Hiatus "
                        "– match units of the input data: "
                    ))
                    break
                except ValueError:
                    print("⚠️  Invalid number. Please enter a decimal or integer value.")
        
        
        
        self.IBIS_intro = """
                 ==============    ==========     =============   ============
                        =          =         =          =         =
                        =          =          =         =         =
                        =          =          =         =         =
                        =          =         =          =         =
                        =          ==========           =         ============
                        =          =         =          =                    =
                        =          =          =         =                    =
                        =          =          =         =                    =
                        =          =         =          =                    =
                  =============    ==========      ============   ============

                Integrated Bayesian model for joint estimation of Initial thorium 
                correction and age-depth model for Speleothems
                
                """

        self.IBIS_BIRD = """             
                                                          ===
                                                        ========
                                                       ===========
                                                      =========  =======
                                                        ==            =========
                                                         ==                 =======
                                                          =                       ====
                                                           ==                         === 
                                                            ===                         ==
                                                             ===                         =
                                                            ====
                                            ====    ===   ===== 
                                       =====    ===       =====                         
                                   ====                      =
                               ====                        ======
                            =                               === 
                      =======                           ====
                   ====                              ===
                =====                              ===   
               =                ========== =  =
                ===  ============         =     =   
                 ===========           =        =
                                    =           =
                                  =             =
                                 =              =
                                =               =
                               =                =
                               =                =
                              =                 =
                             =                    =
                            =                        =
                        =====                           =====
                    ===============                   ================
                    
            """
        if show_bird is True:
            print(self.IBIS_intro) 
            print(self.IBIS_BIRD)

    def Get_Top_Age(self): 
        if not self.Top_Age_Stal: 
            raise("Not applicable")
        if not isinstance(self.collect_data, str):
            raise TypeError("collect_data should be a string")  # Ensure it's a string
        try:
            collection_date = datetime.strptime(self.collect_data, "%d-%m-%Y")
            today = datetime.now()
            age = today.year - collection_date.year - ((today.month, today.day) < (collection_date.month, collection_date.day))
            return age
        except ValueError as e:
            raise ValueError(f"Date format error: {e}")  # Provide feedback on what went wrong


    def Get_IBIS_Input_Data(self): 
        Ibis_input = self.df_reduced

        return Ibis_input

    def Setup_Bounds_and_Uncertainties(self): 
        if os.path.exists(self.bounds_file_name):
            # If the file exists, load the bounds and uncertainties
            with open(self.bounds_file_name, 'rb') as input:
                self.bounds_params = pickle.load(input)
                print('Bounds and uncertainties file exists and is loaded.')
        else:
            # Initialize bounds and uncertainties
            # These bounds 0
            r08 = self.df_reduced['Th230_238U_ratios'].values
            r08_err = self.df_reduced['Th230_238U_ratios_err'].values
            r28 = self.df_reduced['Th232_238U_ratios'].values
            r28_err = self.df_reduced['Th232_238U_ratios_err'].values
            r48 = self.df_reduced['U234_U238_ratios'].values
            r48_err = self.df_reduced['U234_U238_ratios_err'].values
            
            self.bounds_ibis_ext = IBIS_Bounds_and_Uncertainties.IBIS_bounds_and_Uncertainties(r08, r28, r48,
            r08_err, r28_err, r48_err, self.bounds_file_name, results_folder = self.output_dir)

            
            self.bounds_ibis_ext.save_bounds()  # Ensure this method computes and saves the data to the file
            path = os.path.join(self.output_dir, self.bounds_file_name)
            with open(path, 'rb') as f:
                self.bounds_params = pickle.load(f)
                print('Bounds and uncertainties computed and saved.')

        # Extract necessary bounds and uncertainties for further analysis
        self.test_ages = self.bounds_params[0]
        self.age_max = self.bounds_params[2]
        self.age_uncertainties = self.bounds_params[1]
        self.are_there_bounds = True
        

    def Get_IBIS_Bounds(self): 
        if not self.are_there_bounds: 
            self.Setup_Bounds_and_Uncertainties()
        
        return self.test_ages, self.age_max, self.age_uncertainties   

    def Initialize_Thoth(self):
        # build the full path to the .pkl you want
        prior_fname = f"{self.kdes_name}.pkl"
        prior_path = os.path.join(self.output_dir, prior_fname)

        if os.path.exists(prior_path):
            # load it
            with open(prior_path, 'rb') as f:
                self.Thor_prior = pickle.load(f)
            print(f"♻️  Loaded existing Thorium prior from\n   {prior_path}")

        else:
            # compute & save via your IBIS_Thoth_Robust helper
            self.thoth = IBIS_Thoth_Exploration.IBIS_Thoth_Robust(
                self.df_reduced,
                self.age_max,
                num_samples=5000,
                file_name=self.kdes_name,
                results_folder=self.output_dir
            )
            # this must write to output_dir/file_name.pkl
            self.thoth.save_thor_prior()

            # now re‑load it
            with open(prior_path, 'rb') as f:
                self.Thor_prior = pickle.load(f)
            print(f"✅  Computed & saved Thorium prior to\n   {prior_path}")

        self.thor_kde = self.Thor_prior
        self.Thor_Prior = True


    def Set_Up_MCMC(self): 
        """
        Need to 
        """
        if not self.are_there_bounds: 
            self.Setup_Bounds_and_Uncertainties()

        if not self.Thorium_prior_exist: 
            self.Initialize_Thoth()
        
        return self.thor_kde

    

    def Look_At_Initialization(self): 
        self.Ibis_Chains = IBIS_MCMC_Initial_Th.IBIS_MCMC(self.thor_kde,
                                              self.age_max,
                                              self.age_uncertainties,
                                              self.df_reduced,
                                              iterations = self.MCMC_samples,
                                              burn_in = self.MCMC_burn_in,
                                              sample_name = self.sample_name,
                                              n_chains = self.n_chains,
                                              Start_from_pickles = self.Start_from_pickles, 
                                              Include_Detrital = self.Include_Detrital)

        Ini_guess = self.Ibis_Chains.Initial_Guesses_for_Model()
        
        return Ini_guess
        
    def Generate_samples_from_Prior(self):
        # Helper function to get bounds for the prior plot
        # Everything will look nicer
        from scipy.interpolate import interp1d

        x_min = 0
        x_max = 500
        grid_points = 10000
        x_grid = np.linspace(x_min, x_max, grid_points)
        pdf_vals = self.thor_kde(x_grid)

        # Compute the cumulative distribution function (CDF) using the trapezoidal rule.
        cdf_vals = np.cumsum(pdf_vals)
        # Normalize the CDF
        cdf_vals = cdf_vals / cdf_vals[-1]
        #Build an inverse CDF (quantile function) interpolator.
        inv_cdf = interp1d(cdf_vals,
        x_grid, bounds_error=False,
        fill_value=(x_min, x_max))
    
        # Generate uniform random samples between 0 and 1, and invert them.
        u = np.random.rand(100000)
        samples = inv_cdf(u)
        
        return samples
        
    def Plot_Priors(self): 
        if not self.Thorium_prior_exist: 
            self.Initialize_Thoth()

        """
        Plot Check
        -----------
        """
        samples = self.Generate_samples_from_Prior()
        
        low_p, high_p = np.percentile(samples,
                                    [0.001, 99],
                                    axis = 0)
                                    
        print(high_p)
                                    
        x_eval = np.linspace(low_p, high_p, 100000)
        densities = self.thor_kde(x_eval)
        from scipy.ndimage import gaussian_filter1d
        densities_smoothed = gaussian_filter1d(densities, sigma = 70)

        fig, ax = plt.subplots(1,1 , figsize = (5, 5))
        ax.plot(x_eval, densities_smoothed,
              lw = 1.6,
              alpha = 1, 
              color = 'navy', 
               label = 'Speleothem Prior')

        ax.fill_between(x_eval, densities_smoothed,
                        alpha = 0.4,
                      color = 'dodgerblue')

        ax.set_ylabel('Density')
        ax.set_xlabel('$^{230}$Th/$^{232}$Th initial')
        ax.set_xlim(0, high_p)
        #ax.set_ylim(0, ymax + 0.01)  # Extend y-limit slightly above the max density for visual clarity
        ax.legend(ncol = 5,
                 fontsize = 12, 
                 labelspacing = 1.5)


    def Look_at_initial(self): 
        if not self.set_up_the_chain:
            self.thor_kde = self.Set_Up_MCMC()

        self.Ibis_Chains = IBIS_MCMC_Initial_Th.IBIS_MCMC(self.thor_kde,
                                              self.age_max,
                                              self.age_uncertainties,
                                              self.df_reduced,
                                              iterations = self.MCMC_samples,
                                              burn_in = self.MCMC_burn_in,
                                              sample_name = self.sample_name,
                                              n_chains = self.n_chains, 
                                              Start_from_pickles = self.Start_from_pickles, 
                                              Include_Detrital = self.Include_Detrital)

        return self.Ibis_Chains.Initial_Guesses_for_Model()
                
        

    def Run_MCMC(self): 
        if not self.set_up_the_chain: 
            self.thor_kde = self.Set_Up_MCMC()

        self.Ibis_Chains = IBIS_MCMC_Initial_Th.IBIS_MCMC(self.thor_kde,
                                              self.age_max,
                                              self.age_uncertainties,
                                              self.df_reduced,
                                              iterations = self.MCMC_samples,
                                              burn_in = self.MCMC_burn_in,
                                              sample_name = self.sample_name,
                                              results_folder = self.output_dir,
                                              n_chains = self.n_chains,
                                              Start_from_pickles = self.Start_from_pickles)

        self.Ibis_Chains.Run_MCMC();
        self.Chain_run = True




    def Get_MCMC_Results(self): 

        self.Run_MCMC()
        self.results_dicts = self.Ibis_Chains.Get_Results_Dictionary()


    def Get_Post_Vals(self): 
        if not self.Chain_run: 
            self.Run_MCMC()
            self.results_dicts  = self.Ibis_Chains.Get_Results_Dictionary()
        self.Ibis_Chains.Get_Posterior_Values()
        

    def Posterior_plot(self):         
        
        if not self.Chain_run: 
            self.Run_MCMC()
            self.results_dicts  = self.Ibis_Chains.Get_Results_Dictionary()
        self.Ibis_Chains.Get_Posterior_plot()


    def Model_U_ages(self): 
        if not self.Chain_run: 
            self.Run_MCMC()
            self.results_dicts  = self.Ibis_Chains.Get_Results_Dictionary()


        self.U_series_ages, self.U_series_ages_err = self.Ibis_Chains.Get_Useries_Ages()

        return self.U_series_ages, self.U_series_ages_err 




    def Model_Initial_Thorium(self): 
        if not self.Chain_run: 
            self.Run_MCMC()
            self.results_dicts = self.Ibis_Chains.Get_Results_Dictionary()


        self.Model_Ini_Thorium, self.Model_Ini_Thorium_err = self.Ibis_Chains.Get_Initial_Thoriums()

        return self.Model_Ini_Thorium, self.Model_Ini_Thorium_err


    def Model_Initial_U234(self): 
        if not self.Chain_run: 
            self.Run_MCMC()
            self.results_dicts = self.Ibis_Chains.Get_Results_Dictionary()


        self.Model_Ini_Thorium, self.Model_Ini_Thorium_err = self.Ibis_Chains.Get_234U_initial()

        return self.Model_Ini_Thorium, self.Model_Ini_Thorium_err



    def Plot_U_ages_and_Age_model(self, AGE_MODEL = True): 
        self.model_ages, self.model_ages_err, self.model_depths, self.model_depths_err = self.Model_Ages_Depths()
        self.U_series_ages, self.U_series_ages_err  = self.Model_U_ages()

        
        fig, ax = plt.subplots(1,1, figsize = (5,7))
        ax.errorbar(x = self.U_series_ages,
            xerr = self.U_series_ages_err[0], 
                y = self.df_reduced['Depths'].values,
                fmt = 'o',
               label = 'U Series\nModel', 
               alpha = 1, 
                markerfacecolor = 'dodgerblue',
            ecolor = 'k',
           markersize = 10, 
           markeredgecolor = 'k')

        if AGE_MODEL:
        
            ax.fill_betweenx(self.model_depths, self.model_ages - self.model_ages_err[0], 
                       self.model_ages_err[1] + self.model_ages,
                       label = 'Age-Depth\nModel', 
                       alpha = 0.5)
        
            ax.set_ylabel('Depth (mm)')
            
        
        ax.set_xlabel('Apparent Age (a)')
        ax.set_ylim(self.df_reduced['Depths'].values.max() + 5, 
                    self.df_reduced['Depths'].values.min() - 5)
        
        ax.legend(loc = 3)


    def Save_All(self): 
        self.Ibis_Chains.Save_Initial_Thorium()
        print(f'Initial thorium model data saved to {self.sample_name}_Initial_Thoriums.xlsx')
        self.Ibis_Chains.Save_Useries_Ages()
        print(f'Useries age model data saved to {self.sample_name}_U_Series_Ages.xlsx')
        self.Ibis_Chains.Save_234U_Initial()
        print(f'Initial 234U saved to {self.sample_name}_Initial_234U.xlsx')
        

    def Get_Chain_Stats_Thor(self): 
        return self.Ibis_Chains.In_Thor_Chain_Stats()


    def Get_In_Thor(self): 
        return self.Ibis_Chains.Get_Initial_Thoriums()
        
        
    def Get_Chain_Stats_Uages(self): 
        return self.Ibis_Chains.Useries_Age_Chain_Stats()

    def Get_Chain_Stats_Lam_U234(self): 
        return self.Ibis_Chains.lam234_Chain_Stats()

    def Get_Chain_Stats_Lam_Th230(self): 
        return self.Ibis_Chains.Th230_Chain_Stats()


    def Load_U_Series_Ages(self, filename): 
        df = pd.read_excel(filename)
        U_series_ages = df['U_ages'].values
        U_series_ages_err_low = df['U_Age_low'].values
        U_series_ages_err_high = df['U_Age_high'].values
        # Return ages and a tuple of their error bounds (low, high)
        return U_series_ages, (U_series_ages_err_low, U_series_ages_err_high)     


    

    def Run_MCMC_Strat(self): 
        u_ages_file = f'{self.sample_name}_U_Series_Ages.xlsx'
        import pandas as pd

        if os.path.exists(u_ages_file): 
            print(f"File '{u_ages_file}' exists. Skipping initial MCMC and running stratigraphy MCMC directly. Sit Tight. Time for a cup of tea.")
            self.U_series_ages, self.U_series_ages_err = self.Load_U_Series_Ages(u_ages_file)
        else: 
            print("Bayesian Ages Dont Exist Yet! Running IBIS Part1. Hold on...")
            self.U_series_ages, self.U_series_ages_err = self.Model_U_ages()
    

        U_ages = self.U_series_ages
        U_ages_low = self.U_series_ages_err[0]
        U_ages_high = self.U_series_ages_err[1]

        if self.Hiatus_Check:
            self.Ibis_Stratigraphy = IBIS_strat_Hiatus.IBIS_Strat(U_ages,
                                               U_ages_low,
                                               U_ages_high,
                                               self.df_reduced, 
                                                self.sample_name,
                                               self.Start_from_pickles, 
                                               self.n_chains, 
                                               iterations = self.MCMC_Strat_samples, 
                                               burn_in = int(self.MCMC_Strat_samples)/2,
                                                       Hiatus_start = self.Hiatus_start, 
                                                       Hiatus_end = self.Hiatus_end,
                                                Top_Age_Stal = self.Top_Age_Stal)

            self.Ibis_Stratigraphy.Run_MCMC_Strat();
            self.Chain_strat_run = True

        else: 

            self.Ibis_Stratigraphy = IBIS_strat.IBIS_Strat(U_ages, 
                                               U_ages_low,
                                               U_ages_high,
                                               self.df_reduced, 
                                                self.sample_name,
                                               self.Start_from_pickles, 
                                               self.n_chains, 
                                               iterations = self.MCMC_Strat_samples, 
                                               burn_in = int(self.MCMC_Strat_samples)/2, 
                                               Top_Age_Stal = self.Top_Age_Stal)
            self.Ibis_Stratigraphy.Run_MCMC_Strat();
            self.Chain_strat_run = True

    def Age_Model(self): 
        if not self.Chain_strat_run: 
            self.Run_MCMC_Strat()
            self.results_dicts  = self.Ibis_Stratigraphy.Get_Results_Dictionary()


        self.age_low, self.age_high, self.age_median = self.Ibis_Stratigraphy.Get_Age_Model()

        return self.age_low, self.age_high, self.age_median

    def Depth_Model(self): 
        if not self.Chain_strat_run: 
            self.Run_MCMC_Strat()
            self.results_dicts  = self.Ibis_Stratigraphy.Get_Results_Dictionary()


        self.depth_low, self.depth_high, self.depth_median = self.Ibis_Stratigraphy.Get_Depth_Model()

        return self.depth_low, self.depth_high, self.depth_median

    def Get_Age_Depth_Plot(self): 
        self.Ibis_Stratigraphy.Get_Age_Depth_Plot()

    def Save_Age_Depth_Model(self): 
        self.Ibis_Stratigraphy.Save_Age_Depth_Model()
        print(f'Age/Depth model saved to: {self.sample_name}_Age_Depth_Model.xlsx')
        



