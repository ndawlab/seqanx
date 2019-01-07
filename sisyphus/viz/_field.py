import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def plot_field(ax, cbar=False, viable='0.8', reward='#f3e1db', shock='#1c142a'):
    """Plot cliff-walking environment.
    
    Parameters
    ----------
    ax : matplotlib Axes
        Axes in which to draw the plot.
    cbar : bool
        Whether to draw a colorbar.
    viable : str
        Color of viable squares.
    reward : str
        Color of rewarding tile.
    end : str
        Color of ending tile.
        
    Returns
    -------
    ax : matplotlib Axes
        Axes in which to draw the plot.
    """
    
    ## Define grid.
    grid = np.zeros((11, 11))    # Viable states
    grid[5,[2,-3]] = [1, 2]      # Reward/shock states
    
    ## Define colormap.
    cmap = ListedColormap([viable, reward, shock])

    ## Plot cliff.
    ax = sns.heatmap(grid, cmap=cmap, cbar=cbar)
    ax.set(xticklabels=[], yticklabels=[], title='Cliff Walking')  
    
    ## Add outline.
    ax.vlines(np.arange(1,11),0,11,lw=0.1)
    ax.hlines(np.arange(1,11),0,11,lw=0.1)
    
    return ax