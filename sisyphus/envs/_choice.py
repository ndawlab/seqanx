import numpy as np
from scipy.stats import norm
from ._base import GraphWorld

class FreeChoice(GraphWorld):
    """Instrumental variant of the free choice task.
    
    Parameters
    ----------
    rewards : array
        Outcome values.
    probs : array
        Probability of rewards. Defaults is uniform probability.
    
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
        Pandas DataFramestoring the dynamics of the Markov decision process.
        Rows correspond to each viable Q-value, whereas each column contains
        its associated information.
                
    References
    ----------
    1. Leotti, L. A., & Delgado, M. R. (2011). The inherent reward of choice. 
       Psychological science, 22(10), 1310-1318.
    2. Leotti, L. A., & Delgado, M. R. (2014). The value of exercising control 
       over monetary gains and losses. Psychological science, 25(2), 596-604.
    """
    
    def __init__(self, rewards=[-1,1], probs=None):
    
        ## Error-catching.
        if probs is None: probs = np.ones_like(rewards) / len(rewards)
        assert len(rewards) == len(probs)
        
        ## Define one-step transition matrix.
        n = len(rewards)
        T = np.zeros((n+5,n+5)) * np.nan
        T[0,[1,2]] = 1                            # First choice
        T[1,[3,4]] = 1                            # Second choice
        T[2:5,-n:] = 1                            # Reward transitions
        T[np.arange(5,n+5),np.arange(5,n+5)] = 1  # Terminal states

        ## Define rewards.
        R = np.copy(T)
        R[np.where(~np.isnan(R))] = 0
        R[2:5,-n:] = rewards                      # Reward transitions
        
        ## Define start/terminal states.
        start = 0
        terminal = np.arange(5,n+5)

        ## Initialize GraphWorld.
        GraphWorld.__init__(self, T, R, start, terminal, epsilon=0)
        
        ## Update info.
        for s in [2,3,4]:
            ix, = np.where(self.info.S == s)
            self.info = self.info.drop(ix[1:]).reset_index(drop=True)
            self.info.at[ix[0],'T'] = probs
        
    def __repr__(self):
        return '<GraphWorld | Instrumental Free Choice>'
    
    def _draw_nodes(self, ax, s, free_color, fixed_color):
    
        ## Draw initial choice.
        xpos = [0.0]; ypos = [0.0]
        ax.plot(xpos, ypos, marker='o', markersize=s, fillstyle='top', linestyle='none',
                color=free_color, markerfacecoloralt=fixed_color, markeredgecolor='k')

        ## Draw free choice.
        xpos = [1.0, 2.0, 2.0]; ypos = [0.25, 0.5, 0.0]
        ax.plot(xpos, ypos, marker='o', markersize=s, fillstyle='full', linestyle='none',
                color=free_color, markerfacecoloralt=fixed_color, markeredgecolor='k')

        ## Draw fixed choice.
        xpos = [1.0, 2.0]; ypos = [-0.25, -0.5]
        ax.plot(xpos, ypos, marker='o', markersize=s, fillstyle='full', linestyle='none',
                color=fixed_color, markerfacecoloralt=free_color, markeredgecolor='k')

        return ax

    def _draw_edges(self, ax):

        ## Draw edges.
        ax.plot([0.0, 2.0], [0.0, 0.5], color='k', zorder=0)
        ax.plot([0.0, 2.0], [0.0,-0.5], color='k', zorder=0)
        ax.plot([1.0, 2.0], [0.25,0.0], color='k', zorder=0)

        return ax

    def _draw_edge_labels(self, ax, fontsize=14):

        ax.text(0.5,  0.15, '0', ha='center', va='bottom', fontsize=fontsize)
        ax.text(0.5, -0.15, '0', ha='center', va='top', fontsize=fontsize)
        ax.text(1.5,  0.35, r'$[-1, 1]$', ha='center', va='bottom', fontsize=fontsize, rotation=15)
        ax.text(1.5,  0.15, r'$[-1, 1]$', ha='center', va='top', fontsize=fontsize, rotation=-15)
        ax.text(1.5, -0.35, r'$[-1, 1]$', ha='center', va='top', fontsize=fontsize, rotation=-15)
        return ax

    def plot_free_choice(self, ax, s=50, free_color='#834c7d', fixed_color='0.9',
                         edge_labels=False):
        """Plot free choice environment.

        Parameters
        ----------
        ax : matplotlib Axes
            Axes in which to draw the plot.
        s : float (default = 50)
            The marker size in points.
        choice_color : str
            Marker color for free choice nodes.
        fixed_color : str
            Marker color for fixed choice nodes.
        edge_labels : bool
            Draw edge labels.

        Returns
        -------
        ax : matplotlib Axes
            Axes in which to draw the plot.
        """

        import matplotlib.pyplot as plt
        import seaborn as sns
        
        ## Draw DAG.
        ax = self._draw_edges(ax)
        ax = self._draw_nodes(ax, s=50, free_color=free_color, fixed_color=fixed_color)

        ## Optional details.
        if edge_labels: ax = self._draw_edge_labels(ax, fontsize=18)

        ## Clean up.
        ax.set(xlim=(-0.5,2.5), xticks=[], ylim=(-0.7,0.7), yticks=[])
        sns.despine(top=True, right=True, bottom=True, left=True, ax=ax)   

        return ax