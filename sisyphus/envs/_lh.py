import numpy as np
from ._base import GraphWorld, grid_to_adj

class Helplessness(GraphWorld):
    
    def __init__(self, outcomes=[10,-10,5,0], epsilon=0):
        
        ## Define gridworld.
        self.grid = np.arange(5*21).reshape(5,21)
        self.shape = self.grid.shape
        
        ## Define start/terminal states.
        start = 56
        terminal = np.array([44,48,52,60])

        ## Define one-step transition matrix.
        T = grid_to_adj(self.grid, terminal)
        self.T = T
        
        ## Define rewards.
        R = np.zeros_like(T) 
        for s, r in zip(terminal, outcomes): R[:,s] = r
        R[terminal,terminal] = 0
        R *= T
        
        ## Initialize GraphWorld.
        GraphWorld.__init__(self, T, R, start, terminal, epsilon)
        self.R = R
        
    def __repr__(self):
        return '<GraphWorld | Learned Helplessness>'