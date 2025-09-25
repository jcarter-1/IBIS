import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


class IbisProxy:
    """
    Class for constructing
    a continuous proxy curve
    through time
    -------------------------
    Input
    =======
    age, age_uncertainty
    proxy, proxy_uncertainty
    depth, depth_uncertainty
    
    Output
    =======
    Interpolated envelope encompassing uncertainties in both proxy and age.
    
    Method
    ======
    PchipInterpolator
    Uses mon
    """
    
    def __init__(self, proxy, proxy_err,
                 age, age_err):
    
        self.age = age
        self.age_err = age_err
        
        self.proxy = proxy
        self.proxy_err = proxy_err
        
        # Uncertainties output from IBIS are usually asymmetric.
        # The degree of assymetry is variable
        # for Monte Carlo it is far more straightforward to deal with
        # symmetric uncertainties so we simple average out the two
        # sides.
        self.age_err_symm_est = (self.age_err[1] - self.age_err[0]) / 2

        
    
    
    
        
