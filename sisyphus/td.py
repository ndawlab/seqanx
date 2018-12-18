import numpy as np
from .utilities import check_params, softmax

"""Temporal difference submodule."""

class Agent(object):
        
    def __init__(self, beta, eta, gamma, tau):
        
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.tau = tau
        check_params(beta=beta, eta=eta, gamma=gamma, tau=tau)

    def _select_action(Q, beta=1):
        """Select next action (simulation).

        Parameters
        ----------
        Q : array, shape = (n_states,)
            State-action values.

        Returns
        -------
        a : int
            Next action.
        """    

        ## Compute likelihood of actions under policy.
        theta = _softmax(Q * beta)

        ## Select next action.
        return np.searchsorted(theta.cumsum(), np.random.random())       
        
    def run_episode(self, env, Q, start=None, n_steps=100):
        '''Run a single test episode (i.e. Q-values not updated).
        
        Parameters
        ----------
        env : GridWorld instance
            Simulation environment.
        Q : array, shape = (n_states, 4)
            State-action values.
        start : int
            Starting state. Defaults to environment.
        n_steps : int
            Maximum number of steps in trial.
            
        Returns
        -------
        Q : array, shape = (n_states, 4)
            Updated state-action values.
        '''   

        ## Define starting state.
        if start is None: start = env.start
        s = start  

        for _ in np.arange(n_steps):

            ## Check for termination.
            s = int(states[-1])
            if s in env.terminal: break

            ## Select action.
            a = select_action(Q[s], self.beta)
                        
            ## Observe next state and reward.
            s_prime = env.T[s,a]
            r = env.R[s,a]

            ## Update model.
            Q = self._update_model(Q, s, a, r, s_prime)
            
            ## Update state.
            s = s_prime

        return Q
        
class OffPolicy(Agent):
    '''Q-learning agent.
    
    Parameters
    ----------
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
    
    def __init__(self, beta, eta, gamma):
        
        Agent.__init__(self, beta, eta, gamma, None)
        self.params = dict(beta=beta, eta=eta, gamma=gamma)
                
    def __repr__(self):
        return '<OffPolicy | beta = {0}, eta = {1}, gamma = {2}>'.format(*self.params.values())
    
    def _update_model(self, Q, s, a, r, s_prime):
        
        ## Find best action in successor state.
        a_prime = np.argmax(Q[s_prime])
        
        ## Compute reward prediction error.
        delta = r + self.gamma * Q[s_prime,a_prime] - Q[s,a]
        
        ## Update model.
        Q[s,a] += self.eta * delta
        
        return Q

class ExpValSARSA(object):
    '''(Off-policy) expected value SARSA.
    
    Parameters
    ----------
    beta : float
      Inverse temperature for choice.
    eta : float
      Learning rate.
    gamma : float
      Discount factor.
    tau : float
      Inverse temperature for learning.
      
    Notes
    -----
    Expected value SARSA is a variant of the classic SARSA algorithm that is expected
    to have slightly faster convergence by weighing the value of the successor state,
    s', by the likelihood of its respective actions under the current policy. As 
    discussed in Sutton & Barto, the learning rule is as follows:
    
    .. math::
    
        \delta = r + \gamma \sum_a \pi(a' | s')Q(s',a') - Q(s,a)
        
    Note that this is similar to including a softmax function in the learning rule. Thus,
    there are several possible regimes given :math:`\beta` and :math:`\tau`.

    - :math:`\tau >> 0`: Q-learning (Sutton & Barto, 1998).
    - :math:`\tau = \beta`: expected value SARSA (Sutton & Barto, 1998).
    - :math:`\tau > 0`: soft Q-learning (Nachum et al., 2017).
    - :math:`\tau = 0`: stochastic learning.
    - :math:`\tau < 0`: beta-pessimistic learning (Gaskett, 2003).
    - :math:`\tau << 0`: minimax learning (Heger, 1994).
    
    References
    ----------
    1. Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction. MIT press.
    2. Heger, M. (1994). Consideration of risk in reinforcement learning. In Machine Learning 
       Proceedings 1994 (pp. 105-111).
    3. Gaskett, C. (2003). Reinforcement learning under circumstances beyond its control.
    4. Nachum, O., Norouzi, M., Xu, K., & Schuurmans, D. (2017). Bridging the gap between value 
       and policy based reinforcement learning. In Advances in Neural Information Processing 
       Systems (pp. 2775-2785).
    '''
    
    def __init__(self, beta, eta, gamma, tau):
        
        Agent.__init__(self, beta, eta, gamma, tau)
        self.params = dict(beta=beta, eta=eta, gamma=gamma, tau=tau)
                
    def __repr__(self):
        return '<SARSA | beta = {0}, eta = {1}, gamma = {2}, tau = {3}>'.format(*self.params.values())
    
    def _update_model(self, Q, s, a, r, s_prime):
        
        ## Compute likelihood of action.
        theta = softmax(Q[s_prime], self.tau)
        
        ## Compute reward prediction error.
        delta = r + self.gamma * np.sum(Q[s_prime] * theta) - Q[s,a]
        
        ## Update model.
        Q[s,a] += self.eta * delta
        
        return Q