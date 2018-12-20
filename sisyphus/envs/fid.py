import numpy as np
from pandas import DataFrame
from scipy.stats import norm

class FlightInitiationDistance(object):
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
        
        ## Define initial state.
        start = 0
        
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
        R[:n,n] = np.linspace(1,6,n)         # Safety transition
        R[:n,n+1] = -10                      # Danger transition
        R[[n,n+1],[n,n+1]] = 0               # Terminal states 
        
        ## Define start / terminal states.
        self.start = start
        self.terminal = []
        for s in range(T.shape[0]):
            s_prime, = np.nonzero(~np.isnan(T[s]))
            if np.all(s_prime == s): self.terminal.append(s)

        ## Define state information.
        self.states = np.arange(T.shape[0])
        self.n_states = self.states.size

        self.viable_states = self.states[~np.in1d(self.states, self.terminal)]
        self.n_viable_states = self.viable_states.size

        ## Compute transition probabilities.
        states = np.arange(runway)
        cdf = norm(mu, sd).cdf(states)
        
        ## Iteratively define MDP information.
        info = []
        for s, s_prime in np.array(np.where(T==1)).T:
            
            Q = dict()
            
            ## Store state information.
            Q["S"] = s
            Q["S'"] = np.append(s_prime, np.where(T[s] == 0)).astype(int)
            
            ## Store reward information.
            Q["R"] = R[Q["S"], Q["S'"]]
            
            ## Store transition information.
            p = cdf[states==s]
            Q["T"] = np.append(1-p, np.ones(Q["R"].size-1)*p)
            
            info.append(Q)
        
        ## Store MDP information.
        self.info = DataFrame(info, columns=("S","S'","R","T"))
            
    def __repr__(self):
        return '<GridWorld | Flight Initiation Distance>'