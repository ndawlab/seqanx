import numpy as np
import matplotlib.pyplot as plt
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