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
    
    def __init__(self, runway=10, mu=5, sd=1, shock=-1):
    
        
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
        R[:n,n] = np.arange(n) + 1           # Safety transition
        R[:n,n+1] = shock                    # Danger transition
        R[[n,n+1],[n,n+1]] = 0               # Terminal states 
        
        ## Define start/terminal states.
        start = 0
        terminal = []
        for s in range(T.shape[0]):
            s_prime, = np.nonzero(~np.isnan(T[s]))
            if np.all(s_prime == s): terminal.append(s)

        ## Initialize GraphWorld.
        GraphWorld.__init__(self, T, R, start, terminal, 0)
            
        ## Remove masochistic Q-values (i.e. agent cannot elect to be eaten).
        sane_ix = [False if arr[0]==shock else True for arr in self.info['R']]
        self.info = self.info[sane_ix].reset_index(drop=True)
            
        ## Update probability of being eaten.
        states = np.arange(runway)
        cdf = norm(mu, sd).cdf(states)
        
        for i, row in self.info.iterrows():
    
            s, s_prime = row["S"], row["S'"][0]
            if not s_prime in self.terminal:
                self.info.at[i,'T'] = np.array([1-cdf[s], 0, cdf[s]])
                
    def __repr__(self):
        return '<GraphWorld | Flight Initiation Distance>'