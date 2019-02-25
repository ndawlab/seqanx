"""Dynamic programming module"""

import numpy as np
from copy import deepcopy
from ._misc import check_params, softmax, pessimism
from warnings import warn

class ValueIteration(object):
    """Q-value iteration algorithm.
    
    Parameters
    ----------
    policy : max | min | softmax | pessimism (default = pessimism)
        Learning rule.
    gamma : float (default = 0.9)
        Temporal discounting factor.
    beta : float (default = 10.0)
        Inverse temperature for future choice (ignored if policy not softmax).
    w : float (default = 1.0)
        Pessimism weight (ignored if policy not pessimism).
    tol : float, default: 1e-4
        Tolerance for stopping criteria.
    max_iter : int, default: 100
        Maximum number of iterations taken for the solvers to converge.

    References
    ----------
    1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
    """
    
    def __init__(self, policy='pessimism', gamma=0.9, beta=10.0, w=1.0, tol=0.0001, max_iter=100):

        ## Define choice policy.
        self.policy = policy
        if policy == 'max': self._policy = np.max
        elif policy == 'min': self._policy = np.min
        elif policy == 'softmax': self._policy = lambda arr: arr @ softmax(arr * self.beta)
        elif policy == 'pessimism': self._policy = lambda arr: pessimism(arr, self.w)
        else: raise ValueError('Policy "%s" not valid!' %self.policy)
        
        ## Check parameters.
        self.gamma = gamma
        self.beta = beta
        self.w = w
        check_params(gamma=self.gamma, beta=self.beta, w=self.w)
        
        ## Set convergence criteria.
        self.tol = tol
        self.max_iter = max_iter
        
    def __repr__(self):
        return '<Q-value iteration>'
            
    def copy(self):
        """Return copy of agent."""
        return deepcopy(self)
        
    def _q_solve(self, info, Q=None):
        """Solve for Q-values iteratively."""
        
        ## Initialize Q-values.
        if Q is None: Q = np.zeros(info.shape[0], dtype=float)
        assert np.equal(Q.shape, info.shape[0])
        copy = info.copy()
            
        ## Main loop.
        for k in range(self.max_iter):
            
            ## Make copy.
            q = Q.copy()
            
            ## Precompute successor value. 
            copy['Q'] = q
            V_prime = copy.groupby('S').Q.apply(self._policy).values

            ## Compute Q-values.
            for i in range(info.shape[0]):
                                        
                ## Update Q-value.
                Q[i] = sum(info.loc[i,"T"] * (info.loc[i,"R"] + self.gamma * V_prime[info.loc[i,"S'"]]))

            ## Compute delta.
            delta = np.abs(Q - q)

            ## Check for termination.
            if np.all(delta < self.tol): break
           
        return Q, k + 1
    
    def _v_solve(self, info):
        """Compute state value from Q-table."""
        
        ## Copy info and append Q-values.
        copy = info.copy()
        copy['Q'] = self.Q
        
        ## Identify max by state.
        return copy.groupby('S').Q.max().values
        
    def _pi_solve(self, gym):
        """Compute policy from Q-table."""
        
        ## Precompute optimal q(s,a).
        copy = gym.info.copy()
        copy['Q'] = self.Q
        copy = copy.iloc[copy.groupby('S').Q.idxmax().values]
        copy["S'"] = copy["S'"].apply(lambda arr: arr[0])
        
        ## Initialize policy from initial state.
        policy = [gym.start]
        
        ## Iterately append.
        while True:

            ## Termination check.
            s = policy[-1]
            if s in gym.terminal: break
                
            ## Observe successor.
            s_prime, = copy.loc[copy["S"]==s, "S'"].values
            
            ## Terminate on loops. Otherwise append.
            if s_prime in policy: break
            policy.append(s_prime)
                
        return policy
            
    def fit(self, gym, Q=None, verbose=True):        
        """Solve for optimal policy.
        
        Parameters
        ----------
        gym : GridWorld instance
            Simulation environment.
            
        Returns
        -------
        self : returns an instance of self.
        """
        
        ## Solve for Q-values.
        self.Q, self.n_iter = self._q_solve(gym.info, Q)
        if np.equal(self.n_iter, self.max_iter) and verbose:
            warn('Reached maximum iterations.')
        
        ## Solve for values.
        self.V = self._v_solve(gym.info)
        
        ## Compute policy.
        self.pi = self._pi_solve(gym)
                
        return self