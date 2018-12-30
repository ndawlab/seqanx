"""Temporal difference module"""

import numpy as np
from .misc import check_params, softmax, betamax, categorical

def greedy_choice(arr, epsilon):
    if np.random.binomial(1,1-epsilon): 
        return np.argmax(arr)
    else: 
        return np.random.choice(np.arange(len(arr)),1,replace=False)
    
def soft_choice(arr, beta):
    theta = softmax(arr * beta)
    return categorical(theta)  

class ModelFree(object):
    '''Q-learning agent.
    
    Parameters
    ----------
    policy : max | min | softmax | betamax (default = softmax)
        Choice policy.
    beta : float
      Inverse temperature for choice.
    eta : float
      Learning rate.
    gamma : float
      Discount factor.

    References
    ----------
    1. Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction. MIT press.
    '''
    
    def __init__(self, policy, beta=10, eta=0.2, gamma=0.9):
        
        ## Define choice policy.
        self.policy = policy
        if policy == 'max': self._policy = np.max
        elif policy == 'min': self._policy = np.min
        elif policy == 'softmax': self._policy = lambda arr: arr @ softmax(arr * self.beta)
        elif policy == 'betamax': self._policy = lambda arr: betamax(arr, self.beta)
        else: raise ValueError('Policy "%s" not valid!' %self.policy)
        
        ## Check parameters.
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        check_params(beta=self.beta, eta=self.eta, gamma=self.gamma)       
              
    def __repr__(self):
        return '<Model Free Agent>'
            
    def _run_episode(self, Q, gym, choice, epsilon, n_steps=100):
        
        ## Define starting state.
        copy = gym.info.copy()
        s = gym.start  

        for _ in np.arange(n_steps):

            ## Check for termination.
            if s in gym.terminal: break
                
            ## Select action.
            copy['Q'] = Q.copy()
            a = copy[copy.S==s].index[choice(copy.loc[copy.S==s,'Q'], epsilon)]
                        
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
            
        Returns
        -------
        self : returns an instance of self.
        '''   
        
        ## Define metadata.
        if choice == 'greedy': 

            ## Define choice rule.
            _choice = greedy_choice

            ## Define schedule.
            if schedule is None: 
                schedule = 0.05 * np.ones(100)
            else: 
                assert np.all(np.logical_and(schedule >= 0, schedule <= 1))
                
        elif choice == 'softmax':

            ## Define choice rule.
            _choice = soft_choice

            ## Define schedule.
            if schedule is None: 
                schedule = 10.0 * np.ones(100)
            else: 
                assert np.all(np.logical_and(schedule >= -50, schedule <= 50))
            
        else: 
            raise ValueError('Choice "%s" not valid!' %choice)
            
        ## Initialize Q-values.
        if not hasattr(self,'Q') or overwrite: 
            Q = np.zeros(gym.info.shape[0])
        else: 
            Q = self.Q.copy()
            
        ## Solve for Q-values.
        for e in schedule: Q = self._run_episode(Q, gym, _choice, e, n_steps)
        self.Q = Q
        
        ## Solve for values.
        self.V = self._v_solve(gym.info)
        
        ## Compute policy.
        self.pi = self._pi_solve(gym)
                
        return self