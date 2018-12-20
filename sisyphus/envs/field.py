import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from .base import GridWorld

class OpenField(GridWorld):
    """Open field task environment.
    
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
    shape : tuple
        Size of gridworld.
    R : array, shape = (n_states,)
        Reward associated with transitioning to given state.
    T : sparse CSR matrix
        One-step transition matrix where row and col indices denote
        denote state and successor state, respectively, and data
        denote the associated Q-value.
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