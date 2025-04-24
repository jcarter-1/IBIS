# Set path to code folder
import sys
sys.path.append('/Users/johncarter/Documents/IBIS Update')
# Import all codes needed
from ibis_codes import USeries_Age_Equations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde, norm, uniform, lognorm
from scipy.interpolate import interp1d
import matplotlib as mpl
import os
import pickle


# In[3]:


class IBIS_bounds_and_Uncertainties: 
    """
    This function produced neccessary bounds and uncertaintis for 
    the IBIS model 
    -------------------------------------------------------------

    Returns
    -------
    - Age uncertainties (w/ decay constant uncertainties)
    - Maximum age of the speleothem
    - Minimum and Maximum Thorium bounds
    """
    def __init__(self, r08, r48, r08_err, r48_err, bounds_filename):
        self.r08 = r08
        self.r08_err = r08_err
        self.r48 = r48
        self.r48_err = r48_err
        self.lam_230 = 9.1577e-6
        self.lam_230_err = 1.3914e-8
        self.lam_234 = 2.8263e-6
        self.lam_234_err = 2.8234e-9
        self.max_age = None
        self.uncertainties = None
        self.Check_Bounds_and_uncertainties = False # Initially flag as fall untill all bounds and uncertainties are calculated
        self.ages = None
        self.Bounds_ = None
        self.bounds_filename = bounds_filename
        

        self.lam_230 = 9.1577e-6
        self.lam_230_err = 1.3914e-8
        self.lam_234 = 2.8263e-6
        self.lam_234_err = 2.8234e-9
        self.max_age = None
        self.uncertainties = None
        self.Check_Bounds_and_uncertainties = False # Initially flag as fall untill all bounds and uncertainties are calculated
        self.ages = None
        self.Bounds_ = None
        self.bounds_filename = bounds_filename


    def Age_Calc_Analytical_w_decay_const(self): 
        ages = []
        uncertainties = []
        for i in range(len(self.r08)): 
            
    
            lam_230_ = np.array([self.lam_230, self.lam_230_err])
            lam_234_ = np.array([self.lam_234, self.lam_234_err])
        
            lam_diff = lam_230_[0] - lam_234_[0]
            
            Age_class = USeries_Age_Equations.USeries_ages(self.r08[i], self.r48[i], lam_230_[0], 
                                                          lam_234_[0])

            Age = Age_class.Age_solver(age_guess = 1e4)
            
            k1 = (lam_230_[0]/lam_diff) * ( 1 - np.exp(-lam_diff * Age))
        
            k3 = Age * np.exp(-lam_230_[0] * Age) - ((self.r48[i] - 1)/lam_diff) * ((lam_234_[0]/lam_diff) * (1 - np.exp(-lam_diff * Age) - lam_230_[0] * Age * np.exp(-lam_diff * Age)))
        
           
            k4 = (lam_230_[0]/lam_diff) *(self.r48[i] - 1) *(((1-np.exp(-lam_diff * Age))/lam_diff) - Age * np.exp(-lam_diff * Age))
            
            D = lam_230_[0] * (np.exp(-lam_230_[0] * Age) + (self.r48[i] - 1)* np.exp(-lam_diff * Age))
        
            
        
            sigma_age2 = (self.r08_err[i]**2 +(k1**2 * self.r48_err[i]**2) +  k4**2 * lam_234_[1]**2)/(D**2)
            ages.append(Age)
            uncertainties.append(np.sqrt(sigma_age2))

        self.uncertainties = np.array(uncertainties)
        self.ages = np.array(ages)
        return self.ages, self.uncertainties

    def Maximum_Age(self):
        if self.ages is not None:
            self.max_age = self.ages[-1] + 3*self.uncertainties[-1]
        return self.max_age

    def Get_Bounds(self): 
        # Make sure we have all the bounds
        self.ages, self.uncertainties= self.Age_Calc_Analytical_w_decay_const()
        self.max_age = self.Maximum_Age()
        self.Bounds_ = (self.ages, self.uncertainties, self.max_age)



    def save_bounds(self): 
        if self.Bounds_ is None: 
            self.Get_Bounds()

        with open(self.bounds_filename, 'wb') as f: 
            pickle.dump(self.Bounds_, f)
            print(f'Ages, Unceratinties, and Maximum age saved to {self.bounds_filename}')
            
        return self.Bounds_

    def load_bounds(self): 
        if os.path.exists(self.bounds_filename): 
            with open(self.bounds_filename, 'rb') as f: 
                self.Bounds_ = pickle.load(f)
                print('Bounds_loaded from file')
        else: 
            print('Bounds do not exist yet, generating...')
            self.save_bounds()
        




