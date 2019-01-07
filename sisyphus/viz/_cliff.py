import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def plot_cliff(ax, cbar=False, viable='0.8', cliff='0.1', start='#1f77b4', end='#2ca02c'):
    """Plot cliff-walking environment.
    
    Parameters
    ----------
    ax : matplotlib Axes
        Axes in which to draw the plot.
    cbar : bool
        Whether to draw a colorbar.
    viable : str
        Color of viable squares.
    cliff : str
        Color of cliff edge.
    start : str (deprecated)
        Color of starting tile.
    end : str (deprecated)
        Color of ending tile.
        
    Returns
    -------
    ax : matplotlib Axes
        Axes in which to draw the plot.
    """
    
    ## Define grid.
    grid = np.zeros((11, 12))    # Viable states
    grid[-1, 1:-1] = 1           # Cliff edge
    
    ## Define colormap.
    cmap = ListedColormap([viable, cliff])
    
    ## Alternate coloring.
    # grid[-1, 0] = 2              # Start
    # grid[-1,-1] = 3              # Finish
    # cmap = ListedColormap([viable, cliff, start, end])

    ## Plot cliff.
    ax = sns.heatmap(grid, cmap=cmap, cbar=cbar)
    ax.set(xticklabels=[], yticklabels=[], title='Cliff Walking')  
    
    ## Add outline.
    ax.vlines(np.arange(1,12),0,10,lw=0.1)
    ax.hlines(np.arange(1,11),0,12,lw=0.1)
    
    ## Add text.
    ax.annotate('S', (0, 0), (0.041, 0.04), 'axes fraction', ha='center', 
                va='center', fontsize=18, weight='semibold')
    ax.annotate('G', (0, 0), (0.954, 0.04), 'axes fraction', ha='center', 
                va='center', fontsize=18, weight='semibold')
    
    return ax