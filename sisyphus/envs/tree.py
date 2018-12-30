import numpy as np
from scipy.stats import norm
from .base import GraphWorld

class DecisionTree(GraphWorld):
    """Variant of the decision tree game.
    
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
    1. Huys, Q. J., Eshel, N., O'Nions, E., Sheridan, L., Dayan, P., & Roiser, J. P. (2012). 
       Bonsai trees in your head: how the Pavlovian system sculpts goal-directed choices by 
       pruning decision trees. PLoS computational biology, 8(3), e1002410.
    2. Lally, N., Huys, Q. J., Eshel, N., Faulkner, P., Dayan, P., & Roiser, J. P. (2017). 
       The neural basis of aversive Pavlovian guidance during planning. 
       Journal of Neuroscience, 0085-17.
    """
    
    def __init__(self):
        
        ## Define one-step transition matrix.
        T = np.ones((15,15)) * np.nan
        T[0,[ 1, 2]] = 1
        T[1,[ 3, 4]] = 1
        T[2,[ 5, 6]] = 1
        T[3,[ 7, 8]] = 1
        T[4,[ 9,10]] = 1
        T[5,[11,12]] = 1
        T[6,[13,14]] = 1
        T[np.arange(7,15),np.arange(7,15)] = 1

        ## Define rewards.
        R = np.copy(T)
        R[np.where(~np.isnan(R))] = 0
        R[0,[ 1, 2]] = [-70,-20]
        R[1,[ 3, 4]] = [-20,-70]
        R[2,[ 5, 6]] = [-20,-70]
        R[3,[ 7, 8]] = [-20, 20]
        R[4,[ 9,10]] = [ 20,140]
        R[5,[11,12]] = [-20, 20]
        R[6,[13,14]] = [-20, 20]
        
        ## Define start/terminal states.
        start = 0
        terminal = np.arange(7,15)

        ## Initialize GraphWorld.
        GraphWorld.__init__(self, T, R, start, terminal, 0)
        
    def __repr__(self):
        return '<GraphWorld | Decision Tree>'