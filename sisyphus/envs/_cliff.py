import numpy as np
from ._base import GraphWorld, grid_to_adj

class CliffWalking(GraphWorld):
    """Cliff-walking task environment.
    
    References
    ----------
    1. Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction. MIT press.
    2. Gaskett, C. (2003). Reinforcement learning under circumstances beyond its control.
    """
    
    def __init__(self, epsilon=0):
    
        ## Define gridworld.
        self.grid = np.arange(11 * 12, dtype=int).reshape(11,12)
        self.shape = self.grid.shape

        ## Define start/terminal states.
        start = 120
        terminal = np.array([121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131])

        ## Define one-step transition matrix.
        T = grid_to_adj(self.grid, terminal)
        
        ## Define rewards.
        R = -1 * np.ones_like(T)              # Majority transitions
        R[:,terminal[:-1]] = -100             # Cliff transitions
        R[:,terminal[-1]] = 0                 # Safety transitions
        R[terminal,terminal] = 0              # Terminal transitions
        R *= T
            
        ## Initialize GraphWorld.
        GraphWorld.__init__(self, T, R, start, terminal, epsilon)
        
    def __repr__(self):
        return '<GraphWorld | Cliff-Walking Task>'
    
    def plot_cliff(self, annot=True, grid_color='0.8', cliff_color='0.1', start_color='0.8', 
                   goal_color='0.8', cbar=False, annot_kws=None, ax=None):
        """Plot cliff-walking environment.

        Parameters
        ----------
        annot : bool
            Annotate states.
        grid_color : str
            Color of grid tiles.
        cliff_color : str
            Color of cliff tiles.
        start_color : str
            Color of starting tile.
        goal_color : str
            Color of goal tile.
        cbar : bool
            Whether to draw a colorbar.
        annot_kws : dict of key, value mappings, optional
            Keyword arguments for ax.text when annot is True. 
        ax : matplotlib Axes
            Axes in which to draw the plot.

        Returns
        -------
        ax : matplotlib Axes
            Axes in which to draw the plot.
        """

        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import ListedColormap
        
        ## Initialize canvas.
        if ax is None: fig, ax = plt.subplots(1,1,figsize=(5,5))
        
        ## Define grid.
        grid = np.zeros((11, 12))    # Grid titles
        grid[-1, 1:-1] = 1           # Cliff edge
        grid[-1,0] = 2               # Start tile
        grid[-1,-1] = 3              # Goal tile

        ## Define colormap.
        cmap = ListedColormap([grid_color, cliff_color, start_color, goal_color])

        ## Plot cliff.
        ax = sns.heatmap(grid, cmap=cmap, cbar=cbar)
        ax.set(xticklabels=[], yticklabels=[])  

        ## Add outline.
        ax.vlines(np.arange(1,12),0,10,lw=0.1)
        ax.hlines(np.arange(1,11),0,12,lw=0.1)

        ## Annotate.
        if annot:
            if annot_kws is None: annot_kws = dict()
            ax.text(0.5,10.5,'S',ha='center',va='center',**annot_kws)
            ax.text(11.5,10.5,'G',ha='center',va='center',**annot_kws)

        return ax
    
    def plot_policy(self, ax, pi, color='w', head_width=0.25, head_length=0.25):
        """Plot agent policy on grid world.

        Parameters
        ----------
        ax : matplotlib Axes
            Axes in which to draw the plot.
        pi : array
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

        ## Iteratively plot arrows.
        for i in range(len(pi)-1):

            ## Identify S, S' coordinates.
            y1, x1 = np.where(self.grid==pi[i])
            y2, x2 = np.where(self.grid==pi[i+1])

            ## Plot.
            ax.arrow(int(x1)+0.5, int(y1)+0.5, 0.5*int(x2-x1), 0.5*int(y2-y1), 
                     color=color, head_width=head_width, head_length=head_length)

        return ax