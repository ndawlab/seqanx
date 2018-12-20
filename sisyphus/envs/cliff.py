import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from .base import GridWorld

class CliffWalking(GridWorld):
    """Cliff-walking task environment.
    
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