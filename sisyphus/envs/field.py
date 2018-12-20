import numpy as np
from pandas import DataFrame
from scipy.spatial.distance import cdist

class OpenField(object):
    """Open field task environment."""
    
    def __init__(self, epsilon=0):
    
        ## Define gridworld.
        grid = np.ones((11,11), dtype=int)
        self.shape = grid.shape

        ## Define start/terminal states.
        start = 5
        terminal = np.array([57,63])

        ## Define grid adjacency matrix.
        nx, ny = self.shape
        rr = np.array(np.meshgrid(np.arange(nx),np.arange(ny)))
        rr = rr.reshape(2,np.product(self.shape),order='F').T
        A = (cdist(rr,rr)==1).astype(int)
        
        ## Define one-step transition matrix.
        T = np.where(A, 1, np.nan)            # Non-terminal transitions
        T[terminal] = np.nan                  # Terminal transitions
        T[terminal,terminal] = 1              # Terminal transitions

        ## Define rewards.
        R = -1 * np.ones_like(T)              # Majority transitions
        R[:,57] =  100                        # Reward transition
        R[:,63] = -100                        # Punishment transition
        R[terminal,terminal] = 0              # Terminal transitions
        R *= T

        ## Define start / terminal states.
        self.start = start
        self.terminal = terminal
            
        ## Define state information.
        self.states = np.arange(grid.size).reshape(self.shape)
        self.n_states = self.states.size
        
        viable_states = np.logical_xor(grid.flatten(), np.isin(self.states.flatten(), self.terminal))
        self.viable_states = np.argwhere(viable_states).squeeze()
        self.n_viable_states = self.viable_states.size

        ## Iteratively define MDP information.
        info = []
        for s in range(self.n_states):
            
            ## Observe information.
            s_prime, = np.where(~np.isnan(T[s]))
            r = R[s, s_prime]
            t = np.append(1-epsilon, np.ones(r.size-1)*epsilon)
            
            ## Iteratively append.
            for i in range(s_prime.size): 
                info.append({ "S":s, "S'":s_prime, "R":r, "T":np.roll(t,i) })
        
        ## Store.
        self.info = DataFrame(info, columns=("S","S'","R","T"))
        
    def __repr__(self):
        return '<GridWorld | Open Field Task>'