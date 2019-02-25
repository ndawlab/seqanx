"""Miscellaneous functions"""

import numpy as np
from numba import jit
from warnings import warn

@jit
def softmax(arr):
    """Scale-robust softmax choice rule."""
    arr = np.exp(arr - np.max(arr))
    return arr / arr.sum()

@jit
def pessimism(arr, w):
    """Pessimistic learning rule."""
    return w * np.max(arr) + (1 - w) * np.min(arr)

def categorical(arr):
    """Categorical distribution rng."""
    return np.argmax(np.random.multinomial(1,arr))

def check_params(beta=None, eta=None, gamma=None, w=None, epsilon=None):
    """Internal convenience function for sanity checking parameter values."""
    if beta is not None and abs(beta) > 50: 
        warn('Parameter "beta" set very large.')
    if eta is not None and (eta < 0 or eta > 1): 
        raise ValueError('Parameter "eta" must be in range [0,1].')        
    if gamma is not None and (gamma < 0 or gamma > 1): 
        raise ValueError('Parameter "gamma" must be in range [0,1].') 
    if w is not None and (w < 0 or w > 1): 
        raise ValueError('Parameter "w" must be in range [0,1].')       
    if epsilon is not None and (epsilon < 0 or epsilon > 1): 
        raise ValueError('Parameter "epsilon" must be in range [0,1].') 