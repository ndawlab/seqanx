import numpy as np
from ._base import GraphWorld, grid_to_adj

class OpenField(GraphWorld):
    """Open field task environment.
    
    Parameters
    ----------
    reward : float
        Value of reward.
    punishment : float
        Value of punishment.
    
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
    info : DataFrame
        Pandas DataFrame storing the dynamics of the Markov decision process.
        Rows correspond to each viable Q-value, whereas each column contains
        its associated information.
    
    """
    
    def __init__(self, reward=10, punishment=-10):
    
        ## Define gridworld.
        self.grid = np.arange(11 * 11, dtype=int).reshape(11,11)
        self.shape = self.grid.shape

        ## Define start/terminal states.
        start = 115
        terminal = np.array([13,19])

        ## Define one-step transition matrix.
        T = grid_to_adj(self.grid, terminal)

        ## Define rewards.
        R = 0 * np.ones_like(T)               # Majority transitions
        R[:,13] = reward                      # Reward transition
        R[:,19] = punishment                  # Punishment transition
        R[terminal,terminal] = 0              # Terminal transitions
        R *= T

        ## Initialize GridWorld.
        GraphWorld.__init__(self, T, R, start, terminal, epsilon=0)
        
    def __repr__(self):
        return '<GraphWorld | Open Field Task>'
    
    def plot_field(self, reward=10, punishment=-10, annot=True, grid_color='0.8',  
                   reward_color='#f3e1db', punishment_color='#1c142a', 
                   cbar=False, annot_kws=None, ax=None):
        """Plot open field environment.

        Parameters
        ----------
        reward : float
            Reward value.
        punishment : float
            Punishment value.
        annot : bool
            Annotate states.
        grid_color : str
            Color of grid tiles.
        reward_color : str
            Color of rewarding tile.
        punishment_color : str
            Color of punishing tile.
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
        grid = np.zeros((11, 11))    # Viable states
        grid[1,[2,-3]] = [1, 2]      # Reward/punishment states

        ## Define colormap.
        cmap = ListedColormap([grid_color, reward_color, punishment_color])

        ## Plot open field.
        ax = sns.heatmap(grid, cmap=cmap, cbar=cbar, ax=ax)
        ax.set(xticklabels=[], yticklabels=[])  

        ## Add outline.
        ax.vlines(np.arange(1,11),0,11,lw=0.1)
        ax.hlines(np.arange(1,11),0,11,lw=0.1)

        ## Annotate.
        if annot:
            if annot_kws is None: annot_kws = dict()
            ax.text(2.5,1.5,reward,ha='center',va='center',**annot_kws)
            ax.text(8.45,1.5,punishment,ha='center',va='center',**annot_kws)
        
        return ax
    
    def plot_policy(self, ax, pi, color='w', head_width=0.25, head_length=0.25):
        """Plot agent policy on grid world.

        Parameters
        ----------
        ax : matplotlib Axes
            Axes in which to draw the plot.
        pi : array
            Agent policy, i.e. ordered visitation of states.
        color : str, list
            Color(s) of arrow.
        head_width : float (default=0.25)
            Width of the arrow head.
        head_length : float (default=0.25)
            Length of the arrow head.

        Returns
        -------
        ax : matplotlib Axes
            Axes in which to draw the plot.
        """

        ## Error-catching.
        if isinstance(color, str):
            color = [color] * len(pi)
            
        ## Iteratively plot arrows.
        for i in range(len(pi)-1):

            ## Identify S, S' coordinates.
            y1, x1 = np.where(self.grid==pi[i])
            y2, x2 = np.where(self.grid==pi[i+1])

            ## Define arrow coordinates.
            x, y = int(x1) + 0.5, int(y1) + 0.5
            dx, dy = 0.5*int(x2-x1), 0.5*int(y2-y1)
            
            ## Plot.
            ax.arrow(x, y, dx, dy, color=color[i], head_width=head_width, head_length=head_length)
            
        return ax