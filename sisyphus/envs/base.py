import numpy as np
from pandas import DataFrame
from scipy.spatial.distance import cdist

class GridWorld(object):
    """Generate gridworld environment.
    
    Parameters
    ----------
    grid : array, shape = (n,m)
        2-D binary array, where 1 denotes occupiable states and 0 otherwise.
    rewards : array, shape = (n,m)
        2-D array denoting the reward for transitioning from S to S'.
    start : int
        Starting state.
    terminal : int | array (Default None).
        Terminal states.
    
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
    shape : tuple
        Size of gridworld.
    R : array, shape = (n_states,)
        Reward associated with transitioning to given state.
    T : sparse CSR matrix
        One-step transition matrix where row and col indices denote
        denote state and successor state, respectively, and data
        denote the associated Q-value.
    """
    
    def __init__(self, grid, rewards, start, terminal=None):
        
        ## Define metadata.
        self.shape = grid.shape
                
        ## Define start / terminal states.
        if terminal is None: self.terminal = []
        elif isinstance(terminal, int): self.terminal = [terminal]
        else: terminal = self.terminal = np.array(terminal)
        self.start = start
            
        ## Define state information.
        self.states = np.arange(grid.size).reshape(self.shape)
        self.n_states = self.states.size
        
        viable_states = np.logical_xor(grid.flatten(), np.isin(self.states.flatten(), self.terminal))
        self.viable_states = np.argwhere(viable_states).squeeze()
        self.n_viable_states = self.viable_states.size
        
        ## Define one-step transition matrix.
        self.T = self._one_step_transition_matrix() 
        
        ## Define rewards.
        self.R = np.copy(rewards).flatten().astype(float)
        
    def _one_step_transition_matrix(self):
        """Returns the sparse CSR one-step transition matrix."""

        ## Define grid coordinates.
        nx, ny = self.shape
        rr = np.array(np.meshgrid(np.arange(nx),np.arange(ny)))
        rr = rr.reshape(2,np.product(self.shape),order='F').T

        ## Compute one-step adjacency matrix.
        A = (cdist(rr,rr)==1).astype(int)
        
        ## Mask terminal states.
        A[self.terminal] = 0
        A[self.terminal, self.terminal] = 1
        
        ## Convert to sparse CSR matrix.
        data = np.arange(A.sum())
        row, col = A.nonzero()        
        return csr_matrix((data, (row,col)), A.shape, dtype=int)

class GraphWorld(object):

    def __init__(self, T, R, start, epsilon=0):

        ## Error-catching.
        assert np.all(np.any(T, axis=1))              # Check for empty rows
        assert np.allclose(np.isnan(T), np.isnan(R))  # Check identical NaNs
        
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
            Q["T"] = np.append(1-epsilon, [epsilon]*(Q["R"].size-1))
            
            info.append(Q)
        
        ## Store.
        self.info = DataFrame(info, columns=("S","S'","R","T"))