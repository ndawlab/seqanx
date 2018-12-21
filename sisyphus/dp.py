"""Dynamic programming code"""

import numpy as np
from .misc import check_params, softmax
from warnings import warn

class ValueIteration(object):
    """Q-value iteration algorithm.
    
    Parameters
    ----------
    policy : max | min | softmax (default = softmax)
        Choice policy.
    beta : float
        Inverse temperature (ignored if policy not softmax).
    gamma : float
        Discount factor.
    tol : float, default: 1e-4
        Tolerance for stopping criteria.
    max_iter : int, default: 100
        Maximum number of iterations taken for the solvers to converge.

    References
    ----------
    1. Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction. MIT press.
    """
    
    def __init__(self, policy='softmax', beta=10, gamma=0.9, tol=0.0001, max_iter=100):

        ## Define choice policy.
        self.policy = policy
        if policy == 'max': self._policy = np.max
        elif policy == 'min': self._policy = np.min
        elif policy == 'softmax': self._policy = lambda arr: arr @ softmax(arr * self.beta)
        else: raise ValueError('Policy "%s" not valid!' %self.policy)
        
        ## Check parameters.
        self.beta = beta
        self.gamma = gamma
        check_params(beta=self.beta, gamma=self.gamma)
        
        ## Set convergence criteria.
        self.tol = tol
        self.max_iter = max_iter        
        
    def _q_solve(self, info):
        
        ## Initialize Q-values.
        Q = np.zeros(info.shape[0], dtype=float)
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
        optimal_ix = copy.groupby('S').Q.transform(max) == copy.Q
        copy = copy[optimal_ix].copy().reset_index(drop=True)
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
            
    def fit(self, gym):        
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
        self.Q, self.n_iter = self._q_solve(gym.info)
        if self.n_iter == self.max_iter: warn('Reached maximum iterations.')
        
        ## Solve for values.
        self.V = self._v_solve(gym.info)
        
        ## Compute policy.
        self.pi = self._pi_solve(gym)
                
        return self