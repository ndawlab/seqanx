import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def plot_policy(ax, gym, pi, color='w', head_width=0.25, head_length=0.25):
    """Plot agent policy on grid world.
    
    Parameters
    ----------
    ax : matplotlib Axes
        Axes in which to draw the plot.
    gym : GridWorld instance
        Simulation environment.
    py : array
        Agent policy, i.e. ordered visitation of states.
    color : str (default = white)
        Color of arrow.
    head_width : float (default=0.25)
        Width of the arrow head.
    head_length : float (default=0.25)
        Length of the arrow head.
        
    Returns
    -------
    ax : matplotlib Axes
        Axes in which to draw the plot.
    """
        
    ## Initialize grid.
    grid = gym.states.reshape(gym.shape)
    
    ## Iteratively plot arrows.
    for i in range(len(pi)-1):
        
        ## Identify S, S' coordinates.
        y1, x1 = np.where(grid==pi[i])
        y2, x2 = np.where(grid==pi[i+1])

        ## Plot.
        ax.arrow(int(x1)+0.5, int(y1)+0.5, 0.5*int(x2-x1), 0.5*int(y2-y1), 
                 color=color, head_width=head_width, head_length=head_length)
        
    return ax

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
    ax.text(0.5,10.5,'S',ha='center',va='center',fontsize=18,weight='semibold')
    ax.text(11.5,10.5,'G',ha='center',va='center',fontsize=18,weight='semibold')
    
    return ax