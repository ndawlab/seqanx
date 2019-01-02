import numpy as np

def test_world():
    """Returns inputs for 4-state GraphWorld."""
   
    ## Define one-step transition matrix.
    T = np.zeros((4,4)) * np.nan
    T[0,1]         = 1              # Start
    T[1,[2,3]]     = 1              # Choice
    T[[2,3],[2,3]] = 1              # Terminal
    
    ## Define rewards.
    R = np.copy(T)
    R[0,1]         = 0              # Start
    R[1,[2,3]]     = [1,-1]         # Choice
    R[[2,3],[2,3]] = 0              # Terminal
    
    ## Define start/terminal states.
    start = 0
    terminal = [2,3]
    
    return T, R, start, terminal