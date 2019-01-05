import numpy as np
from ._base import GraphWorld, grid_to_adj

class CliffWalking(GraphWorld):
    """Cliff-walking task environment.
    
    
    References
    ----------
    1. Gaskett, C. (2003). Reinforcement learning under circumstances beyond its control.
    """
    
    def __init__(self, epsilon=0):
    
        ## Define gridworld.
        self.grid = np.arange(11 * 12, dtype=int).reshape(11,12)
        self.shape = self.grid.shape

        ## Define start/terminal states.
        start = 120
        terminal = np.array([121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131])

        ## Define one-step transition matrix.
        T = grid_to_adj(self.grid, terminal)
        
        ## Define rewards.
        R = -1 * np.ones_like(T)              # Majority transitions
        R[:,terminal[:-1]] = -100             # Cliff transitions
        R[:,terminal[-1]] = 0                 # Safety transitions
        R[terminal,terminal] = 0              # Terminal transitions
        R *= T
            
        ## Initialize GraphWorld.
        GraphWorld.__init__(self, T, R, start, terminal, epsilon)
        
    def __repr__(self):
        return '<GraphWorld | Cliff-Walking Task>'