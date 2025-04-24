import pandas as pd
import numpy as np 
from scipy.optimize import fsolve

class Age_Hellstrom:
    """
    Class for solving the Ludwig age equation
    """
    def __init__(self,
                 data,
                 r02_initial = None,
                 r02_initial_err = None,
                 U238 = None,
                 U238_err = None):
                 
        self.lam230 = 9.157711462015396e-06
        self.lam230_err = 1.3913817124214168e-08
        self.lam232 = 4.933431890106372e-11
        self.lam232_err =4.933431890106372e-13
        self.lam234 = 2.8262881980018157e-06
        self.lam234_err = 2.8234071702770433e-09
        self.lam238 = 1.551254796141587e-10
        self.lam238_err =  8.332053601458739e-14
        
        self.data = data
        self.r48 = self.data['U_234_r'].values.item()
        self.r48_err = self.data['U_234_r_err'].values.item()
        self.r08 = self.data['Th_230_r'].values.item()
        self.r08_err = self.data['Th_230_r_err'].values.item()
        self.r28 = self.data['Th_232_r'].values.item()
        self.r28_err = self.data['Th_232_r_err'].values.item()
        
        if U238 is not None:
            self.rho_08_48 = (((U238_err/U238)**2)/((self.r08_err/self.r08)**2 + (self.r48_err/self.r48)**2))
            self.rho_28_48 = (((U238_err/U238)**2)/((self.r28_err/self.r28)**2 + (self.r48_err/self.r48)**2))
            self.rho_08_28 = (((U238_err/U238)**2)/((self.r08_err/self.r08)**2 + (self.r28_err/self.r28)**2))

        else:
            self.rho_08_48 = 0.0
            self.rho_28_48 = 0.0
            self.rho_08_28 = 0.0

        # 230Th/232Th initial  ratios
        if r02_initial is None:
            self.r02_initial = 0.88
        if r02_initial_err is None:
            self.r02_initial_err = 0.44
            
    
        
        
    def Internal_Age_Uncertainty(self):
        # Function for the calculation of the internal age uncertainty
        # Internal - Uncertainty on the corrected ratios
        # Need Age
        # Need to define and/or get for age uncertainty estimation
        Age = self.Age_solver_Hellstrom()
        lam_diff = self.lam234 - self.lam230
            
        
        err2 = np.zeros(self.data.shape[0])
        for i in range(self.data.shape[0]):
            A = self.r08 - self.r28 * self.r02_initial * np.exp(-self.lam230 * Age)
            B = 1 - np.exp(-self.lam230 * Age)
            C = (self.r48 - 1) * (self.lam230/lam_diff) * (1 -np.exp(lam_diff*Age))
        
            da_dt = self.lam230 * self.r28 * self.r02_initial * np.exp(-self.lam230 * Age)
            db_dt = self.lam230 * np.exp(-self.lam230 * Age)
            dc_dt = -(self.r48 - 1) * self.lam230 * np.exp((self.lam234 - self.lam230) * Age)

            df_dt = da_dt - db_dt + dc_dt
            
            df_dr08 = 1
            df_dr02 = -self.r28 * np.exp(-self.lam230 * Age)
            df_dr28 = -self.r02_initial * np.exp(-self.lam230 * Age)
            df_dr48 = (self.lam230/lam_diff) * (1 - np.exp(lam_diff*Age))
        
        
            dt_dr08 = df_dr08 / df_dt
            dt_dr02 = df_dr02/ df_dt
            dt_dr28 = df_dr28/ df_dt
            dt_dr48 = df_dr48 / df_dt
            
            jacobian = np.array([dt_dr08, dt_dr02,
                                 dt_dr28, dt_dr48])
            cov = np.zeros((4,4))
            cov[0,0] = self.r08_err**2
            cov[1,1] = self.r02_initial_err ** 2
            cov[2,2] = self.r28_err ** 2
            cov[3,3] = self.r48_err**2
            cov[0,3] = cov[3,0] = self.rho_08_48 * self.r08_err * self.r48_err
            cov[0,2] = cov[2,0] = self.rho_08_28 * self.r08_err * self.r28_err
            cov[2,3] = cov[3,2] = self.rho_28_48 * self.r28_err * self.r48_err

            err2[i] = np.dot(jacobian.T, np.dot(cov, jacobian))
        
    
        return np.sqrt(err2)
        
    def External_Age_Uncertainty(self):
        # Function for the calculation of external uncertainty
        # Decay constants for lambda230 and lambda 234

        # Need to define and/or get for age uncertainty estimation
        Age = self.Age_solver_Hellstrom()
        lam_diff = self.lam234 - self.lam230
        
        da_dt = self.lam230 * self.r28 * self.r02_initial * np.exp(-self.lam230 * Age)
        db_dt = self.lam230 * np.exp(-self.lam230 * Age)
        dc_dt = -(self.r48 - 1) * self.lam230 * np.exp((self.lam234 - self.lam230) * Age)

        df_dt = da_dt - db_dt + dc_dt

        # F = A - B + C
        
        da_dlam1 = Age *self.lam230 * self.r28 * self.r02_initial * (-1 * np.exp(-self.lam230 * Age))
        db_dlam1 = -Age * np.exp(-self.lam230 * Age)
        dc_dlam1 = (self.r48 - 1) * ( np.exp(Age *lam_diff) *(self.lam234*(Age*self.lam230 - 1) - Age*self.lam230**2) + self.lam234 /(lam_diff**2))
        
        df_dlam1 = da_dlam1 - db_dlam1 + dc_dlam1
        
        dt_dlam1 = df_dlam1/df_dt
    
        da_dlam2 = 0
        db_dlam2 = 0
        dc_dlam2 =  ((self.lam230 * (np.exp(lam_diff*Age)*(-self.lam234*Age +Age*self.lam230 + 1) -1))/(lam_diff**2)) * (self.r48 - 1)
        df_dlam2 = da_dlam2 - db_dlam2 + dc_dlam2
        dt_dlam2 = df_dlam2/df_dt
        
        age_err2 = np.zeros(self.data.shape[0])
        
        for i in range(self.data.shape[0]):
            jac = np.array([dt_dlam1, dt_dlam2])
            cov = np.zeros((2,2))
            cov[0,0] = self.lam230_err ** 2
            cov[1,1] = self.lam234_err ** 2
            age_err2[i] = np.dot(jac.T, np.dot(cov, jac))
            
        return np.sqrt(age_err2)
        
    def Age_Equation_Hellstrom(self, T):
        
        A = self.r08 - self.r28 * self.r02_initial * np.exp(-self.lam230 * T)
        B = 1 - np.exp(-self.lam230 * T)
        D = self.r48 - 1
        E = self.lam230 / (self.lam234 - self.lam230)
        F = 1 - np.exp(-(self.lam230 - self.lam234)*T)
        C = D * E * F
        return A - B + C
        
    def Age_solver_Hellstrom(self, age_guess=1e4):
        """
        Solve the U-series age equation for a single age estimate.
        """
        func = lambda age: self.Age_Equation_Hellstrom(age)
        t_solution = fsolve(func, age_guess)
        return t_solution[0]
        

        
    def Get_Ages_and_Uncertainties(self):
        Age = self.Age_solver_Hellstrom()
        
        Internal_err = self.Internal_Age_Uncertainty()
        External_err = self.External_Age_Uncertainty()
        
        Combined = np.sqrt(Internal_err**2 + External_err**2)
        
        return Age, Internal_err, Combined
        
    def Initial_U234(self):

        Age, age_int_err, age_ext_err= self.Get_Ages_and_Uncertainties()

        
        u234_initial = 1 + ((self.r48 - 1) * np.exp(self.lam234 * Age))
        
        err2_int = np.zeros(self.data.shape[0])
        
        for i in range(self.data.shape[0]):
            jac = np.array([np.exp(self.lam234 * Age),
            Age * ((self.r48 - 1) * np.exp(self.lam234 * Age)),
            self.lam234 * ((self.r48 - 1) * np.exp(self.lam234 * Age))])
            cov = np.zeros((3,3))
            cov[0,0] = self.r48_err ** 2
            cov[1,1] = self.lam234_err ** 2
            cov[2,2] = age_int_err **2
            err2_int[i] = np.dot(jac, np.dot(cov, jac.T))
            
        err2_ext = np.zeros(self.data.shape[0])
        for i in range(self.data.shape[0]):
            jac = np.array([np.exp(self.lam234 * Age),
            Age * ((self.r48 - 1) * np.exp(self.lam234 * Age)),
            self.lam234 * ((self.r48 - 1) * np.exp(self.lam234 * Age))])
            cov = np.zeros((3,3))
            cov[0,0] = self.r48_err ** 2
            cov[1,1] = self.lam234_err ** 2
            cov[2,2] = age_ext_err **2
            err2_ext[i] = np.dot(jac, np.dot(cov, jac.T))
            
        return u234_initial, np.sqrt(err2_int), np.sqrt(err2_ext)
        
        
        
    def Age_Equation_uncorrected(self, T):

        
        A = self.r08
        B = 1 - np.exp(-self.lam230 * T)
        D = self.r48 - 1
        E = self.lam230 / (self.lam234 - self.lam230)
        F = 1 - np.exp((self.lam234 - self.lam230)*T)
        C = D * E * F
        return A - B + C
        
    def Age_solver_uncorrected(self, age_guess=1e4):
        """
        Solve the U-series age equation for a single age estimate.
        """
        func = lambda age: self.Age_Equation_uncorrected(age)
        t_solution = fsolve(func, age_guess)
        return t_solution[0]
        
        
        
        
    def Internal_Age_Uncertainty_uncorrected(self):
        # Function for the calculation of the internal age uncertainty
        # Internal - Uncertainty on the corrected ratios
        # Need Age
        # Need to define and/or get for age uncertainty estimation
        Age = self.Age_solver_uncorrected()
        lam_diff = self.lam234 - self.lam230
            
        # Get corrected Ratios
        
        err2 = np.zeros(self.data.shape[0])
        for i in range(self.data.shape[0]):
            A = self.r08
            B = 1 - np.exp(-self.lam230 * Age)
            C = (self.r48 - 1) * (self.lam230/lam_diff) * (1 -np.exp(lam_diff*Age))
        
            da_dt = 0
            db_dt = self.lam230 * np.exp(-self.lam230 * Age)
            dc_dt = -(self.r48 - 1) * self.lam230 * np.exp((self.lam234 - self.lam230) * Age)

        
            df_dt = da_dt - db_dt + dc_dt
            df_dr08 = 1
            df_dr48 = (self.lam230/lam_diff) * (1 - np.exp(lam_diff*Age))
        
        
            dt_dr08 = df_dr08 / df_dt
            dt_dr48 = df_dr48 / df_dt
            
            jacobian = np.array([dt_dr08, dt_dr48])
            cov = np.zeros((2,2))
            cov[0,0] = self.r08_err**2
            cov[1,1] = self.r48_err**2
            cov[0,1] = cov[1,0] = self.rho_08_48 * self.r08_err  *self.r48_err
            
            err2[i] = np.dot(jacobian.T, np.dot(cov, jacobian))
        
    
        return np.sqrt(err2)
        
    def External_Age_Uncertainty_uncorrected(self):
        # Function for the calculation of external uncertainty
        # Decay constants for lambda230 and lambda 234
        
        # Need to define and/or get for age uncertainty estimation
        Age = self.Age_solver_uncorrected()
        lam_diff = self.lam234 - self.lam230
        
        A = self.r08
        B = 1 - np.exp(-self.lam230 * Age)
        C = (self.r48 - 1) * (self.lam230/lam_diff) * (1 -np.exp(lam_diff*Age))
        
        da_dt = 0
        db_dt = self.lam230 * np.exp(-self.lam230 * Age)
        dc_dt = -(self.r48 - 1) * self.lam230 * np.exp((self.lam234 - self.lam230) * Age)

        
        df_dt = da_dt - db_dt + dc_dt
        
        # F = A - B + C
        
        da_dlam1 = 0
        db_dlam1 = -Age * np.exp(-self.lam230 * Age)
        dc_dlam1 = (self.r48 - 1) * ( np.exp(Age *lam_diff) *(self.lam234*(Age*self.lam230 - 1) - Age*self.lam230**2) + self.lam234 /(lam_diff**2))
        
        
        df_dlam1 = da_dlam1 - db_dlam1 + dc_dlam1
        
        dt_dlam1 = df_dlam1/df_dt
        
        da_dlam2 = 0
        db_dlam2 = 0
        dc_dlam2 =  ((self.lam230 * (np.exp(lam_diff*Age)*(-self.lam234*Age +Age*self.lam230 + 1) -1))/(lam_diff**2)) * (self.r48 - 1)
        df_dlam2 = da_dlam2 - db_dlam2 + dc_dlam2
        dt_dlam2 = df_dlam2/df_dt
        
        age_err2 = np.zeros(self.data.shape[0])
        
        for i in range(self.data.shape[0]):
            jac = np.array([dt_dlam1, dt_dlam2])
            cov = np.zeros((2,2))
            cov[0,0] = self.lam230_err ** 2
            cov[1,1] = self.lam234_err ** 2
            age_err2[i] = np.dot(jac.T, np.dot(cov, jac))
            
        return np.sqrt(age_err2)
        
        
    def Get_Ages_and_Uncertainties_uncorrected(self):
        Age = self.Age_solver_uncorrected()
        
        Internal_err = self.Internal_Age_Uncertainty_uncorrected()
        External_err = self.External_Age_Uncertainty_uncorrected()
        
        Combined = np.sqrt(Internal_err**2 + External_err**2)
        
        return Age, Internal_err, Combined
        
        
    def Activity_and_Age_dataframe(self):
        import pandas as pd
        age, age_int_err, age_ext_err= self.Get_Ages_and_Uncertainties()
        age_uncorr, age_uncorr_int_err, age_uncorr_ext_err= self.Get_Ages_and_Uncertainties_uncorrected()
        u234_initial, u234_initial_int_err, u234_initial_ext_err = self.Initial_U234()
        

        df = pd.DataFrame({"Th230_U238_act": self.r08,
                            "Th230_U238_act_err": self.r08_err,
                            "Th232_U238_act": self.r28,
                            "Th232_U238_act_err": self.r28_err,
                            "U234_U238_act": self.r48,
                            "U234_U238_act_err": self.r48_err,
                            "Th230_Th232_initial": self.r02_initial,
                            "Th230_Th232_initial_err": self.r02_initial_err,

                            "Age uncorrected (a)" : age_uncorr,
                            "Age uncorrected internal err (a)" : age_uncorr_int_err,
                            "Age uncorrected external err (a)" : age_uncorr_ext_err,

                            "Age corrected (a)": age,
                            "Age corrected internal err" : age_int_err,
                            "Age corrected external err" : age_ext_err,
                            "U234$_{0}$": u234_initial,
                            "U234$_{0}$_internal_err" : u234_initial_int_err,
                            "U234$_{0}$_external_err" : u234_initial_ext_err
                          },
                          index = [0])
                          
        return df

        
