import numpy as np
from scipy.stats import norm
from .base import GraphWorld

class FreeChoice(GraphWorld):
    """Instrumental variant of the Free Choice task.
    
    Parameters
    ----------
    rewards : array
        Outcome values.
    probs : array
        Probability of rewards.
    
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
    1. Leotti, L. A., & Delgado, M. R. (2011). The inherent reward of choice. 
       Psychological science, 22(10), 1310-1318.
    2. Leotti, L. A., & Delgado, M. R. (2014). The value of exercising control 
       over monetary gains and losses. Psychological science, 25(2), 596-604.
    """
    
    def __init__(self, rewards=[-1,0,1], probs=None):
    
        ## Error-catching.
        if probs is None: probs = np.ones_like(rewards) / len(rewards)
        assert len(rewards) == len(probs)
        
        ## Define one-step transition matrix.
        n = len(rewards)
        T = np.zeros((n+5,n+5)) * np.nan
        T[0,[1,2]] = 1                            # First choice
        T[1,[3,4]] = 1                            # Second choice
        T[2:5,-n:] = 1                            # Reward transitions
        T[np.arange(5,n+5),np.arange(5,n+5)] = 1  # Terminal states

        ## Define rewards.
        R = np.copy(T)
        R[np.where(~np.isnan(R))] = 0
        R[2:5,-n:] = rewards                      # Reward transitions
        
        ## Define start/terminal states.
        start = 0
        terminal = np.arange(5,n+5)

        ## Initialize GraphWorld.
        GraphWorld.__init__(self, T, R, start, terminal, 0)
        
        ## Remove duplicate actions.
        self.info = self.info.drop([5,6,8,9,11,12]).reset_index(drop=True)
        
        ## Update transition probabilities.
        for i in [4,5,6]: self.info.at[i,'T'] = probs
        
    def __repr__(self):
        return '<GraphWorld | Instrumental Free Choice>'