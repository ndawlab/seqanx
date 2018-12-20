import numpy as np
from scipy.stats import norm
from .base import GraphWorld

class FlightInitiationDistance(GraphWorld):
    """Flight initiation distance environment. 
    
    Parameters
    ----------
    runway : int
        Number of states in 2-D runway.
    mu : float
        Average state at which predator strikes.
    sd : float
        Deviation around mean.
    
    Attributes
    ----------
    states : array, shape = (n,)
        Indices of states.
    n_states : int
        Total number of states.
    viable_states : array
        Indices of viable states.
    n_viable_states : int
        Number of viable states.
    info : DataFrame
        Pandas DataFrame  storing the dynamics of the Markov decision process.
        Rows correspond to each viable Q-value, whereas each column contains
        its associated information.
                
    References
    ----------
    1. Qi, S., Hassabis, D., Sun, J., Guo, F., Daw, N., & Mobbs, D. (2018). 
       How cognitive and reactive fear circuits optimize escape decisions in humans. 
       Proceedings of the National Academy of Sciences, 115(12), 3186-3191. 
    """
    
    def __init__(self, runway=10, mu=5, sd=1):
        
        ## Define one-step transition matrix.
        n = runway
        T = np.zeros((n+2,n+2)) * np.nan
        T[np.arange(n),np.arange(n)+1] = 1   # Corridor transitions
        T[:n,n] = 1                          # Safety transition
        T[:n,n+1] = 0                        # Danger transition
        T[[n,n+1],[n,n+1]] = 1               # Terminal states 

        ## Define rewards.
        R = np.copy(T)
        R[np.arange(n),np.arange(n)+1] = 0   # Corridor transitions
        R[:n,n] = np.linspace(1,6,n).round(2)# Safety transition
        R[:n,n+1] = -10                      # Danger transition
        R[[n,n+1],[n,n+1]] = 0               # Terminal states 
        
        ## Define initial state.
        start = 0
        
        ## Initialize object.
        GraphWorld.__init__(self, T, R, start)
        
        ## Update transition probabilities.
        s = np.arange(runway)
        cdf = norm(mu, sd).cdf(s)
        for i in range(self.info.shape[0] - 2):
            p, = cdf[self.info.loc[i,'S']==s]
            self.info.at[i,'T'] = np.where(self.info.loc[i,'T'], 1-p, p)
            
    def __repr__(self):
        return '<GraphWorld | Flight Initiation Distance>'