import numpy as np
from .base import GraphWorld, grid_to_adj

class OpenField(GraphWorld):
    """Open field task environment."""
    
    def __init__(self, epsilon=0):
    
        ## Define gridworld.
        self.grid = np.arange(11 * 11, dtype=int).reshape(11,11)
        self.shape = self.grid.shape

        ## Define start/terminal states.
        start = 5
        terminal = np.array([57,63])

        ## Define one-step transition matrix.
        T = grid_to_adj(self.grid, terminal)

        ## Define rewards.
        R = 0 * np.ones_like(T)               # Majority transitions
        R[:,57] =  10                         # Reward transition
        R[:,63] = -10                         # Punishment transition
        R[terminal,terminal] = 0              # Terminal transitions
        R *= T

        ## Initialize GridWorld.
        GraphWorld.__init__(self, T, R, start, terminal, epsilon)
        
    def __repr__(self):
        return '<GraphWorld | Open Field Task>'