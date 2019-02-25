import numpy as np
from scipy.stats import norm
from ._base import GraphWorld

class BART(GraphWorld):
    """Balloon analog risk task environment.
    
    Parameters
    ----------
    pumps : int
        Maximum number of balloon pumps.
    mu : float
        Average state at which balloon pops.
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
        Pandas DataFrame storing the dynamics of the Markov decision process.
        Rows correspond to each viable Q-value, whereas each column contains
        its associated information.
                
    References
    ----------
    1. Lejuez, C. W., Read, J. P., Kahler, C. W., Richards, J. B., Ramsey, S. E., Stuart, 
       G. L., ... & Brown, R. A. (2002). Evaluation of a behavioral measure of risk taking: 
       the Balloon Analogue Risk Task (BART). Journal of Experimental Psychology: Applied, 8(2), 75.
    """
    
    def __init__(self, pumps=10, mu=5, sd=1):
        
        ## Define one-step transition matrix.
        n = pumps
        T = np.zeros((n+2,n+2)) * np.nan
        T[np.arange(n),np.arange(n)+1] = 1   # Corridor transitions
        T[:n,n] = 1                          # Safety transition
        T[:n,n+1] = 0                        # Danger transition
        T[[n,n+1],[n,n+1]] = 1               # Terminal states 

        ## Define rewards.
        R = np.copy(T)
        R[np.arange(n),np.arange(n)+1] = 0   # Corridor transitions
        R[:n,n] = np.arange(n) + 1           # Safety transition
        R[:n,n+1] = -np.arange(n)            # Danger transition
        R[[n,n+1],[n,n+1]] = 0               # Terminal states 
        
        ## Define start/terminal states.
        start = 0
        terminal = []
        for s in range(T.shape[0]):
            s_prime, = np.nonzero(~np.isnan(T[s]))
            if np.all(s_prime == s): terminal.append(s)

        ## Initialize GraphWorld.
        GraphWorld.__init__(self, T, R, start, terminal, epsilon=0)
            
        ## Remove masochistic Q-values (i.e. agent cannot elect to be eaten).
        bps = self.n_states - 1
        sane_ix = [np.logical_or(arr[0]!=bps, arr.size==1) for arr in self.info["S'"].values]
        self.info = self.info[sane_ix].reset_index(drop=True)
            
        ## Update probability of being eaten.
        states = np.arange(pumps)
        cdf = norm(mu, sd).cdf(states)
        
        for i, row in self.info.iterrows():
    
            s, s_prime = row["S"], row["S'"][0]
            if not s_prime in self.terminal:
                self.info.at[i,'T'] = np.array([1-cdf[s], 0, cdf[s]])
                
    def __repr__(self):
        return '<GraphWorld | Balloon Analog Risk Task>'