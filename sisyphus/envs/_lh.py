import numpy as np
from ._base import GraphWorld, grid_to_adj

class Helplessness(GraphWorld):
    
    def __init__(self, outcomes=[10,-10], epsilon=0):
        
        ## Define gridworld.
        self.grid = np.arange(5*15).reshape(5,15)
        self.shape = self.grid.shape
        
        ## Define start/terminal states.
        start = 44
        terminal = np.array([30, 37])

        ## Define one-step transition matrix.
        T = grid_to_adj(self.grid, terminal)
        self.T = T
        
        ## Define rewards.
        R = np.zeros_like(T) 
        for s, r in zip(terminal, outcomes): R[:,s] = r
        R[terminal,terminal] = 0
        R *= T
        
        ## Initialize GraphWorld.
        GraphWorld.__init__(self, T, R, start, terminal, epsilon)
        self.R = R
        
        ## Update start.
        self.terminal = np.append(self.terminal, start)
        self.info = self.info.loc[self.info.S != 44]
        for i in range(4):
            d = {"S":44, "S'":np.roll([29, 43, 44, 59],i), "T":np.array([1,0,0,0]), "R":np.zeros(4)}
            self.info = self.info.append(d, ignore_index=True)
        self.info = self.info.sort_values('S').reset_index(drop=True)
        
    def __repr__(self):
        return '<GraphWorld | Learned Helplessness>'
    
    def plot_lh(self, reward=10, shock=-10, annot=True, grid_color='0.8',  
                   reward_color='#f3e1db', shock_color='#1c142a', 
                   cbar=False, annot_kws=None, ax=None):
        """Plot cliff-walking environment.

        Parameters
        ----------
        reward : float
            Reward value.
        shock : float
            Shock value.
        annot : bool
            Annotate states.
        grid_color : str
            Color of grid tiles.
        reward_color : str
            Color of rewarding tile.
        shock_color : str
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
        grid = np.zeros_like(self.grid)    # Viable states
        grid[[2,2],[0,7]] = [1, 2]         # Reward/shock states

        ## Define colormap.
        cmap = ListedColormap([grid_color, reward_color, shock_color])

        ## Plot environment.
        ax = sns.heatmap(grid, cmap=cmap, cbar=cbar, ax=ax)
        ax.set(xticklabels=[], yticklabels=[])  

        ## Add outline.
        x,y = self.shape
        ax.vlines(np.arange(1,y),0,x,lw=0.1)
        ax.hlines(np.arange(1,x),0,y,lw=0.1)

        ## Annotate.
        if annot:
            if annot_kws is None: annot_kws = dict()
            ax.text(0.5,2.5,reward,ha='center',va='center',**annot_kws)
            ax.text(7.45,2.5,shock,ha='center',va='center',**annot_kws)
            annot_kws['color'] = 'k'
            ax.text(14.5,2.5,'S',ha='center',va='center',**annot_kws)
        
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