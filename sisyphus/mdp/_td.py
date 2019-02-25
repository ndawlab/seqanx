"""Temporal difference module"""

import numpy as np
from copy import deepcopy
from ._misc import check_params, pessimism, categorical
from ._misc import softmax as _softmax

def epsilon_greedy(arr, epsilon):
    """Epsilon-greedy choice rule."""
    if np.random.binomial(1,1-epsilon): return np.argmax(arr)
    else: return np.random.choice(np.arange(len(arr)),1,replace=False)[0]
    
def softmax(arr, beta):
    """Softmax choice rule."""
    theta = _softmax(arr * beta)
    return categorical(theta)  

class ModelFree(object):
    '''Q-learning agent.
    
    Parameters
    ----------
    policy : max | min | softmax | pessimism (default = pessimism)
        Learning rule.
    eta : float (default = 0.1)
        Learning rate.
    gamma : float (default = 0.9)
        Temporal discounting factor.
    beta : float (default = 10.0)
        Inverse temperature for future choice (ignored if policy not softmax).
    w : float (default = 1.0)
        Pessimism weight (ignored if policy not pessimism).
    gamma : float
      Discount factor.

    References
    ----------
    1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
    '''
    
    def __init__(self, policy='pessimism', eta=0.1, gamma=0.9, beta=10.0, w=1.0):
        
        ## Define choice policy.
        self.policy = policy
        if policy == 'max': self._policy = np.max
        elif policy == 'min': self._policy = np.min
        elif policy == 'softmax': self._policy = lambda arr: arr @ _softmax(arr * self.beta)
        elif policy == 'pessimism': self._policy = lambda arr: pessimism(arr, self.w)
        else: raise ValueError('Policy "%s" not valid!' %self.policy)
        
        ## Check parameters.
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.w = w
        check_params(beta=self.beta, eta=self.eta, gamma=self.gamma, w=self.w)       
              
    def __repr__(self):
        return '<Model Free Agent>'
            
    def copy(self):
        """Return copy of agent."""
        return deepcopy(self)
        
    def _run_episode(self, Q, gym, choice, epsilon, n_steps=100):
        """Run single episode of training."""
        
        ## Define starting state.
        copy = gym.info.copy()
        s = gym.start  

        for _ in np.arange(n_steps):

            ## Check for termination.
            if s in gym.terminal: break
                
            ## Select action.
            copy['Q'] = Q.copy()
            i = choice(copy.loc[copy.S==s,'Q'].values, epsilon)
            a = copy[copy.S==s].index[i]
                        
            ## Observe next state and reward.
            i = categorical(copy.loc[a,'T'])
            s_prime = copy.loc[a,"S'"][i]
            r = copy.loc[a,'R'][i]

            ## Update model.
            v_prime = self._policy(copy.loc[copy.S==s_prime,'Q'])
            delta = r + self.gamma * v_prime - Q[a]
            Q[a] += self.eta * delta
            
            ## Update state.
            s = s_prime

        return Q
    
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
        
    def fit(self, gym, choice='softmax', schedule=None, n_steps=100, overwrite=False):
        '''Run a single test episode (i.e. Q-values not updated).
        
        Parameters
        ----------
        gym : GraphWorld instance
            Simulation environment.
        choice : greedy | softmax
            Choice rule.
        schedule : array
            Parameter value for choice rule (e.g. inverse temperature, epsilon greedy)
            for a particular trial.
        n_steps : int
            Maximum number of steps allowed in a single episode.
        overwrite : True | False
            If true, overwrite previously stored Q-values (if any).
            
        Returns
        -------
        self : returns an instance of self.
        '''   
        
        ## Define metadata.
        if choice == 'greedy': 

            ## Define choice rule.
            choice = epsilon_greedy

            ## Define schedule.
            if schedule is None:  schedule = 0.05 * np.ones(100)
            elif isinstance(schedule, (int, float)): schedule = np.array([schedule])
            assert np.all(np.logical_and(schedule >= 0, schedule <= 1))
                
        elif choice == 'softmax':

            ## Define choice rule.
            choice = softmax

            ## Define schedule.
            if schedule is None: schedule = 10.0 * np.ones(100)
            elif isinstance(schedule, (int, float)): schedule = np.array([schedule])
            assert np.all(np.logical_and(schedule >= -50, schedule <= 50))
            
        else: 
            raise ValueError('Choice "%s" not valid!' %choice)
            
        ## Initialize Q-values.
        if not hasattr(self,'Q') or overwrite: 
            Q = np.zeros(gym.info.shape[0])
        else: 
            Q = self.Q.copy()
            
        ## Solve for Q-values.
        for e in schedule: Q = self._run_episode(Q, gym, choice, e, n_steps)
        self.Q = Q
        
        ## Solve for values.
        self.V = self._v_solve(gym.info)
        
        ## Compute policy.
        self.pi = self._pi_solve(gym)
                
        return self