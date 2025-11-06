import numpy as np
import pandas as pd
import os


class IBIS_Configuration_Class: 
    """
    IBIS configuration class reads in raw measured activity data
    and constructs a refined dataframe for the model input, 
    plot of the input data, constructs priors for the initial thorium. 
    This is a precusor to the model and neccassary for defining parameters and 
    distributions required for the IBIS Bayesian model. 
    """
    def __init__(self, filepath): 
        """
        Initialize the IBIS model with 
        default configurations. 
        """
        self.filepath = filepath
        # Data
        self.Input_Data = None
        self.load_data() 
    
    def load_data(self): 
        """
        Load measured ratios, measured ratio uncertainties, 
        depths, and depth uncertainties
        Parameters: 
        filepath (str) : Path to the file to be read in
        Returns: 
        pd.DataFrame: Dataframe contianing the loaded data
        """
        try:
            self.input_data = pd.read_excel(self.filepath)
            print(f"Data loaded successfully from {self.filepath} (Excel)")
        except Exception as e1:
            try:
                self.input_data = pd.read_csv(self.filepath)
                print(f"Data loaded successfully from {self.filepath} (CSV)")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to read '{self.filepath}' as Excel ({e1}) and as CSV ({e2})"
                )


    def ensure_data_loaded(self):
        if self.input_data is None:
            raise ValueError("Data has not been loaded. Please load data before proceeding.")
            
    def Get_Depths(self):
        self.ensure_data_loaded()
        df = self.input_data
        Depths = df['Depth'].values
        if 'Depth_err' in df.columns:
            Depths_err = df['Depth_err'].values
        else:
            Depths_err = 0.01 * Depths  # Assume a 1% if no error is available
        self.Depths = Depths
        self.Depths_err = Depths_err
        return self.Depths, self.Depths_err
        
    def ensure_depths(self):
        if self.Depths is None or self.Depths_err is None:
            self.Get_Depths()
    
    def Get_Measured_Ratios(self):
        """
        Get Data into the right formatting for uses later on within the model.
        """
        self.ensure_data_loaded()
        Depths, Depths_err = self.Get_Depths()

        # Extracting ratios and their errors
        df = self.input_data
        Th230_238U_ratios = df['Th_230_r'].values
        Th232_238U_ratios = df['Th_232_r'].values
        U234_U238_ratios = df['U_234_r'].values
        Th230_238U_ratios_err = df['Th_230_r_err'].values
        Th232_238U_ratios_err = df['Th_232_r_err'].values
        U234_U238_ratios_err = df['U_234_r_err'].values
        
        # Creaing a DataFrame with the extracted data
        self.df_ratios = pd.DataFrame({
            "Th230_238U_ratios": Th230_238U_ratios,
            "Th230_238U_ratios_err": Th230_238U_ratios_err,
            "Th232_238U_ratios": Th232_238U_ratios,
            "Th232_238U_ratios_err": Th232_238U_ratios_err,
            "U234_U238_ratios": U234_U238_ratios,
            "U234_U238_ratios_err": U234_U238_ratios_err,
            "Depths": Depths,
            "Depths_err": Depths_err
        })

        return self.df_ratios

            





