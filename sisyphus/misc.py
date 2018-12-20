"""Miscellaneous useful code"""

import numpy as np
from numba import jit
from warnings import warn

@jit
def softmax(arr):
    """Scale-robust softmax."""
    arr = np.exp(arr - np.max(arr))
    return arr / arr.sum()

def check_params(beta=None, eta=None, gamma=None, tau=None, epsilon=None):
    
    if beta is not None and abs(beta) > 50: 
        warn('Parameter "beta" set very large.')
    if eta is not None and (eta < 0 or eta > 1): 
        raise ValueError('Parameter "eta" must be in range [0,1].')        
    if gamma is not None and (gamma < 0 or gamma > 1): 
        raise ValueError('Parameter "gamma" must be in range [0,1].') 
    if tau is not None and abs(tau) > 50: 
        warn('Parameter "tau" set very large.')
    if epsilon is not None and (epsilon < 0 or epsilon > 1): 
        raise ValueError('Parameter "epsilon" must be in range [0,1].') 