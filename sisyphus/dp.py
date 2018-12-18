import numpy as np
from .utilities import check_params, argmax, argmin, softmax
from warnings import warn

"""Dynamic programming submodule."""

class ValueIteration(object):
    """Q-value iteration algorithm.
    
    Parameters
    ----------
    policy : max | min | softmax (default = softmax)
        Choice policy.
    transition : fixed | random (default = fixed)
        Currently ignored.
    beta : float
        Inverse temperature (used only if policy is softmax).
    gamma : float
        Discount factor.
    epsilon : float
        State transition randomness.
    tol : float, default: 1e-4
        Tolerance for stopping criteria.
    max_iter : int, default: 100
        Maximum number of iterations taken for the solvers to converge.

    References
    ----------
    1. Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction. MIT press.
    """
    
    def __init__(self, policy='softmax', transition='fixed', beta=10, gamma=0.9, 
                 epsilon=0.01, tol=0.0001, max_iter=100):

        ## Define choice policy.
        self.policy = policy
        if policy == 'max': self._policy = argmax
        elif policy == 'min': self._policy = argmin
        elif policy == 'softmax': self._policy = softmax
        else: raise ValueError('Policy "%s" not valid!' %self.policy)
            
        ## Define transition dynamics.
        self.transition = transition
        if transition not in ['fixed','random']: 
            raise ValueError('Transition "%s" not valid!' %self.policy)
        
        ## Check parameters.
        self.beta = beta
        self.gamma = gamma
        if self.transition == 'fixed': self.epsilon = 0
        else: self.epsilon = epsilon
        check_params(beta=self.beta, gamma=self.gamma, epsilon=self.epsilon)
        
        ## Set convergence criteria.
        self.tol = tol
        self.max_iter = max_iter        
        
    def _compute_value(self, env):
        """Compute state value from Q-table."""
        
        ## Initialize values.
        V = np.ones(env.n_states) * np.nan
        
        ## Iteratively look-up argmax.
        for s in env.viable_states:
            V[s] = np.max(self.Q_[env.T[s].data])
            
        return V
        
    def _compute_policy(self, env):
        """Compute policy from Q-table."""
        
        ## Initialize policy from initial state.
        policy = [env.start]
        
        ## Iterately append.
        while True:

            ## Termination check.
            s = policy[-1]
            if s in env.terminal: break
                
            ## Observe argmax successor.
            qi = np.argmax(self.Q_[env.T[s].data])
            s_prime = env.T[s].indices[qi]
            
            ## Terminate on loops. Otherwise append.
            if s_prime in policy: break
            policy.append(s_prime)
                
        return policy
        
    def _fit(self, env):
        
        ## Initialize Q-values.
        Q = np.zeros(env.T.size, dtype=float)
        
        ## Extract metadata (ignores terminal transitions).        
        Q_index, S, S_prime = [], [], []
        for s in env.viable_states:
            Q_index = np.append(Q_index, env.T[s].data).astype(int)
            S = np.append(S, s * np.ones_like(env.T[s].data)).astype(int)
            S_prime = np.append(S_prime, env.T[s].indices).astype(int)
            
        ## Main loop.
        k = 0
        while k < self.max_iter:
            
            ## Make copy.
            q = Q.copy()

            ## Compute (discounted) expected values.
            dEV = np.zeros_like(Q)
            for i, s_prime in zip(Q_index, S_prime):

                ## Observe reward.
                r = env.R[s_prime]

                ## Observe successor actions.
                q_prime = Q[env.T[s_prime].data]

                ## Compute likelihood of successor actions under policy.
                theta = self._policy(q_prime * self.beta)

                ## Compute expected value.
                dEV[i] = r + self.gamma * (q_prime @ theta)
                
            ## Update Q-values.
            for i, s, s_prime in zip(Q_index, S, S_prime):
                
                ## Compute transition likelihood.
                theta = np.where(env.T[s].indices == s_prime, 1-self.epsilon, self.epsilon)
                
                ## Update.
                Q[i] = dEV[env.T[s].data] @ theta

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