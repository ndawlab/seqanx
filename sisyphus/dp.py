import numpy as np
from numba import jit
from warnings import warn

"""Dynamic programming submodule."""

@jit
def softmax(arr):
    """Scale-robust softmax."""
    arr = np.exp(arr - np.max(arr))
    return arr / arr.sum()

@jit
def argmax(arr):
    onehot = np.zeros_like(arr)
    onehot[np.argmax(arr)] = 1
    return onehot

@jit
def argmin(arr):
    onehot = np.zeros_like(arr)
    onehot[np.argmin(arr)] = 1
    return onehot

class ValueIteration(object):
    """Value iteration algorithm.
    
    Parameters
    ----------
    policy : func1d
        Choice policy. This function should accept 1-D arrays.
    transition : func1d
        Currently ignored.
    beta : float
        Inverse temperature (used only if policy is softmax).
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
    
    def __init__(self, policy, transition=None, beta=10, gamma=0.9, 
                 tol=0.0001, max_iter=100):

        ## Define model.
        self.policy = policy
        
        ## Check parameters.
        if abs(beta) > 50: 
            warn('Parameter "beta" set very large.')
        if gamma < 0 or gamma > 1: 
            raise ValueError('Parameter "gamma" must be in range [0,1].') 

        self.beta = beta
        self.gamma = gamma
        
        ## Set convergence criteria.
        self.tol = tol
        self.max_iter = max_iter
        
    def _compute_value(self, env):
        
        V = np.ones(env.n_states) * np.nan
        for s in env.viable_states:
            V[s] = np.max(self.Q_[env.T[s].data])
        return V
        
    def _compute_policy(self, env):
        
        policy = [env.start]
        while True:

            s = policy[-1]
            if s in env.terminal: break
            qi = np.argmax(self.Q_[env.T[s].data])
            s_prime = env.T[s].indices[qi]
            if s_prime in policy: break
            policy.append(s_prime)
                
        return policy
        
    def _fit(self, env):
        
        ## Initialize Q-values.
        Q = np.zeros(env.T.size, dtype=float)
        
        ## Extract metadata (ignores terminal transitions).
        Q_index = np.concatenate([env.T[s].data for s in env.viable_states])
        S_prime = np.concatenate([env.T[s].indices for s in env.viable_states])
                            
        ## Main loop.
        k = 0
        while k < self.max_iter:
            
            ## Make copy.
            q = Q.copy()

            for i, s_prime in zip(Q_index, S_prime):

                ## Observe reward.
                r = env.R[s_prime]
                
                ## Observe successor actions.
                q_prime = Q[env.T[s_prime].data]
                
                ## Compute likelihood of successor actions under policy.
                theta = self.policy(q_prime * self.beta)
                
                ## Update Q-value.
                Q[i] = r + self.gamma * (q_prime @ theta)

            ## Compute delta.
            delta = np.abs(Q - q)

            ## Check for termination.
            if np.all(delta < self.tol): break
            else: k += 1
                    
        ## Store number of iterations.
        self.n_iter_ = k
        if self.n_iter_ == self.max_iter: warn('Reached maximum iterations.')

        ## Store Q-values.
        self.Q_ = Q
        
        ## Store state values.
        self.V_ = self._compute_value(env)
        
        ## Compute policy.
        self.pi_ = self._compute_policy(env)
                
        return self
            
    def fit(self, env):        
        """Solve for optimal policy.
        
        Parameters
        ----------
        env : GridWorld instance
            Simulation environment.
            
        Returns
        -------
        self : returns an instance of self.
        """
        
        return self._fit(env)