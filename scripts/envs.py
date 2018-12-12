import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

class GridWorld(object):
    """Generate gridworld environment.
    
    Parameters
    ----------
    grid : array, shape = (n,m)
        Binary array where 1 denoting occupiable states and 0 otherwise.
    rewards : array, shape = (n,m)
        Array denoting the reward for transitioning into corresponding state.
    start : int
        Starting state (default = None).
    terminal : list or array
        Terminal states (default = None).
    
    Attributes
    ----------
    states : array, shape = (n)
        List of state indices.
    n_states : int
        Number of unique states.
    viable_states : array
        List of viable state indices
    n_viable_states : int
        Number of viable states.
    shape : tuple
        Size of gridworld.
    R : array, shape = (n_states, 4)
        State-transition rewards.
    T : array, shape = (n_states, 4)
        One-step transition matrix.
    """
    
    def __init__(self, grid, rewards, start=None, terminal=None):
        
        ## Define metadata.
        self.shape = grid.shape
                
        ## Define start / terminal states.
        if terminal is None: self.terminal = []
        elif isinstance(terminal, int): self.terminal = [terminal]
        else: terminal = self.terminal = np.array(terminal)
        self.start = start
            
        ## Define state information.
        self.states = np.arange(grid.size).reshape(self.shape)
        self.n_states = self.states.size
        
        viable_states = np.logical_xor(grid.flatten(), np.isin(self.states.flatten(), self.terminal))
        self.viable_states = np.argwhere(viable_states).squeeze()
        self.n_viable_states = self.viable_states.size
        
        ## Define one-step transition matrix.
        self.T = self._one_step_transition_matrix() 
        
        ## Define reward matrix.
        self.R = rewards.flatten()[self.T]
        self.R[self.terminal] = 0
        
    def _one_step_transition_matrix(self):
        """Returns the one-step transition matrix for the cardinal directions."""

        ## Define grid coordinates.
        nx, ny = self.shape
        rr = np.array(np.meshgrid(np.arange(nx),np.arange(ny)))
        rr = rr.reshape(2,np.product(self.shape),order='F').T

        ## Compute adjacency matrix.
        A = (cdist(rr,rr)==1).astype(int)
        
        ## Construct transition matrix.
        T = np.repeat(np.arange(self.n_states),4).reshape(self.n_states,4)
        for s in self.viable_states:
            s_prime, = np.where(A[s])
            T[s,:s_prime.size] = s_prime
            
        return np.sort(T, axis=-1).astype(int)

class OpenField(GridWorld):
    """Open field task environment.
    
    Attributes
    ----------
    states : array, shape = (n)
        List of state indices.
    n_states : int
        Number of unique states.
    viable_states : array
        List of viable state indices
    n_viable_states : int
        Number of viable states.
    shape : tuple
        Size of gridworld.
    R : array, shape = (n_states, 4)
        State-transition rewards.
    T : array, shape = (n_states, 4)
        One-step transition matrix.
    """
    
    def __init__(self):
    
        ## Define gridworld.
        grid = np.ones((11,11), dtype=int)

        ## Define start/terminal states.
        start = 5
        terminal = np.array([57,63])

        ## Define rewards.
        rewards = np.ones_like(grid) * -1
        rewards[5,2] = 100
        rewards[5,-3] = -100
        
        ## Initialize object.
        GridWorld.__init__(self, grid, rewards, start, terminal)
        
    def __repr__(self):
        return '<GridWorld | Open Field Task>'
        
    def plot_field(self, grid_color='0.8', reward_color='#2ca02c', shock_color='0.1', ax=None):
        """Visualize the open field environment.
        
        Parameters
        ----------
        figsize : tuple
            Width and height of figure (inches).
        grid_color : tuple | str
            Color of grid tiles.
        reward_color : tuple | str
            Color of cliff tiles.
        shock_color : tuple | str
            Color of end tile.
            
        Returns
        -------
        ax
        """
        
        ## Prepare map.
        grid = np.zeros_like(self.states)
        grid[5,2] = 1
        grid[5,-3] = 2
        cmap = ListedColormap([grid_color,reward_color,shock_color])

        ## Plot.
        ax = sns.heatmap(grid, cmap=cmap, cbar=False, linewidths=0.01, linecolor='0.75', ax=ax)
        ax.set(xticklabels=[], yticklabels=[])
        
        return ax
    
class CliffWalking(GridWorld):
    """Cliff-walking task environment.
    
    Attributes
    ----------
    states : array, shape = (n)
        List of state indices.
    n_states : int
        Number of unique states.
    viable_states : array
        List of viable state indices
    n_viable_states : int
        Number of viable states.
    shape : tuple
        Size of gridworld.
    R : array, shape = (n_states, 4)
        State-transition rewards.
    T : array, shape = (n_states, 4)
        One-step transition matrix.
    """
    
    def __init__(self):
    
        ## Define gridworld.
        grid = np.ones((11,12), dtype=int)

        ## Define start/terminal states.
        start = 120
        terminal = np.array([121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131])

        ## Define rewards.
        rewards = np.ones(grid.size) * -1
        rewards[terminal[:-1]] = -100
        rewards[terminal[-1]] = 0
        rewards = rewards.reshape(grid.shape)

        ## Initialize object.
        GridWorld.__init__(self, grid, rewards, start, terminal)
        
    def __repr__(self):
        return '<GridWorld | Cliff-Walking Task>'
        
    def plot_cliff(self, grid_color='0.1', cliff_color='0.8', start_color='#1f77b4', 
                   end_color='#2ca02c', ax=None):
        """Visualize the cliff-walking environment.
        
        Parameters
        ----------
        figsize : tuple
            Width and height of figure (inches).
        grid_color : tuple | str
            Color of grid tiles.
        cliff_color : tuple | str
            Color of cliff tiles.
        start_color : tuple | str
            Color of start tile.
        end_color : tuple | str
            Color of end tile.
            
        Returns
        -------
        axis
        """
        
        ## Prepare map.
        grid = np.ones_like(self.states)
        grid[-1,1:-1] = 0
        grid[-1,0] = 2
        grid[-1,-1] = 3
        cmap = ListedColormap([grid_color,cliff_color,start_color,end_color])

        ## Plot.
        ax = sns.heatmap(grid, cmap=cmap, cbar=False, linewidths=0.01, linecolor='0.75', ax=ax)
        ax.set(xticklabels=[], yticklabels=[])
        
        return ax

class TwoStepTask(GridWorld):
    """Two-step task environment.
    
    Attributes
    ----------
    states : array, shape = (n)
        List of state indices.
    n_states : int
        Number of unique states.
    viable_states : array
        List of viable state indices
    n_viable_states : int
        Number of viable states.
    shape : tuple
        Size of gridworld.
    R : array, shape = (n_states, 4)
        State-transition rewards.
    T : array, shape = (n_states, 4)
        One-step transition matrix.
    """
    
    def __init__(self):
            
        ## Define gridworld.
        grid = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                         [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])

        ## Define start/stop information.     
        start = 60
        terminal = np.array([0, 4, 6, 10])

        ## Define rewards.
        rewards = np.ones(grid.size) * -1
        rewards[0] = 50
        rewards[4] = -100
        rewards[6] = 10
        rewards[10] = 0
        rewards = rewards.reshape(grid.shape)
    
        ## Initialize object.
        GridWorld.__init__(self, grid, rewards, start, terminal)
        
    def __repr__(self):
        return '<GridWorld | Two-Step Task>'
        
    def plot_task(self, grid_color='0.8', reward_color='#2ca02c',
                   shock_color='#1f77b4', bg='k', ax=None):
        """Visualize the two-step task environment.
        
        Parameters
        ----------
        grid_color : tuple | str
            Color of grid tiles.
        cliff_color : tuple | str
            Color of cliff tiles.
        start_color : tuple | str
            Color of start tile.
        end_color : tuple | str
            Color of end tile.
            
        Returns
        -------
        axis
        """
        
        ## Prepare map.
        grid = np.zeros(self.n_states)
        grid[self.viable_states] = 1
        grid[[0,6,10]] = 2
        grid[4] = 3
        grid = grid.reshape(self.shape)
        cmap = ListedColormap([bg, grid_color, reward_color, shock_color])

        ## Plot.
        ax = sns.heatmap(grid, cmap=cmap, cbar=False, linewidths=0.01, linecolor=bg, ax=ax)
        ax.set(xticklabels=[], yticklabels=[])
        
        return ax