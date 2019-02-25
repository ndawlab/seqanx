import numpy as np
from pandas import DataFrame
from scipy.spatial.distance import cdist

def grid_to_adj(grid, terminal=False):
    """Convert grid world to adjacency matrix.
    
    Parameters
    ----------
    grid : array, shape (i,j)
        Grid world.
    terminal : array
        List of terminal states.
        
    Returns
    -------
    T : array, shape (n_states, n_states)
        Adjacency matrix of states.
        
    Notes
    -----
    The initial grid can contain any value. Grid states defined as NaNs 
    are treated as nonviable states and excluded from further processing.        
    """
    
    ## Identify coordinates of viable states.
    rr = np.array(np.where(~np.isnan(grid))).T

    ## Compute adjacency matrix.
    A = (cdist(rr,rr)==1).astype(int)

    ## Define one-step transition matrix.
    T = np.where(A, 1, np.nan)
    
    ## Update terminal states.
    if np.any(terminal):
        T[terminal] = np.nan
        T[terminal,terminal] = 1
    
    return T

class GraphWorld(object):
    """Base graph world object.
    
    Parameters
    ----------
    T : array, shape (n_states, n_states)
        Graph adjacency matrix.
    R : array, shape (n_states, n_states)
        One-step reward function.
    start : int
        Starting state.
    terminal : int | list
        Terminal states.
    epsilon : int
        Randomness parameter. If zero, transitions are deterministic.
        
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
    """
    
    def __init__(self, T, R, start, terminal, epsilon=0):
        
        ## Define start / terminal states.
        self.start = start
        self.terminal = terminal
            
        ## Define state information.
        self.states = np.arange(T.shape[0])
        self.n_states = self.states.size

        self.viable_states = self.states[~np.in1d(self.states, self.terminal)]
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
                info.append({ "S":s, "S'":np.roll(s_prime,i), "R":np.roll(r,i), "T":t })
        
        ## Store.
        self.info = DataFrame(info, columns=("S","S'","R","T"))