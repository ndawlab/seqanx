import numpy as np
from ._base import GraphWorld

class Helplessness(GraphWorld):
    
    def __init__(self, rewards=[-1,1]):
        
        ## Error catching.
        assert np.equal(len(rewards), 2)
            
        ## Define one-step transition matrix.
        T = np.zeros((5, 5)) * np.nan
        T[0, [1, 4]] = 1
        T[1, [2, 3]] = 1
        T[np.arange(2,5),np.arange(2,5)] = 1

        ## Define rewards.
        R = np.copy(T)
        R[np.where(~np.isnan(R))] = 0
        R[1,[2, 3]] = rewards
        
        ## Define start/terminal states.
        start = 0
        terminal = np.arange(2,5)
        
        ## Initialize GraphWorld.
        GraphWorld.__init__(self, T, R, start, terminal, 0)
        
    def __repr__(self):
        return '<GraphWorld | Learned Helplessness>'