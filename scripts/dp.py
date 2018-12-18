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
        
    def _compute_policy(self, env):
        
        policy = [env.start]
        for _ in range(50):
            s_prime = env.T[policy[-1]].indices
            v_prime = self.V_[s_prime]
            policy.append( s_prime[np.argmax(v_prime)] )
            if policy[-1] in env.terminal: break
                
        return np.unique(policy)
        
    def _fit(self, env):
        
        ## Initialize values.
        V = np.zeros(env.n_states, dtype=float)

        ## Main loop.
        i = 0
        while i < self.max_iter:

            ## Make copy.
            v = V.copy()

            for s in env.viable_states:

                ## Compute (discounted) expected value.
                dEV = env.T[s].data + self.gamma * V[env.T[s].indices]

                ## Compute likelihood of action under policy.
                theta = self.policy(dEV * self.beta)

                ## Compute new values.
                V[s] = np.sum(theta * dEV)

            ## Compute delta.
            delta = np.abs(V - v)

            ## Check for termination.
            if np.all(delta < self.tol): break
            else: i += 1
                    
        ## Store number of iterations.
        self.n_iter_ = i
        if self.n_iter_ == self.max_iter: warn('Reached maximum iterations.')

        ## Store values.
        self.V_ = V
        
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