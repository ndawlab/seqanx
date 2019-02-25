import numpy as np
from scipy.stats import norm
from ._base import GraphWorld

class DecisionTree(GraphWorld):
    """Decision tree from aversive pruning experiments.
    
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
        Pandas DataFrame  storing the dynamics of the Markov decision process.
        Rows correspond to each viable Q-value, whereas each column contains
        its associated information.
                
    References
    ----------
    1. Huys, Q. J., Eshel, N., O'Nions, E., Sheridan, L., Dayan, P., & Roiser, J. P. (2012). 
       Bonsai trees in your head: how the Pavlovian system sculpts goal-directed choices by 
       pruning decision trees. PLoS computational biology, 8(3), e1002410.
    2. Lally, N., Huys, Q. J., Eshel, N., Faulkner, P., Dayan, P., & Roiser, J. P. (2017). 
       The neural basis of aversive Pavlovian guidance during planning. 
       Journal of Neuroscience, 0085-17.
    """
    
    def __init__(self):
        
        ## Define one-step transition matrix.
        T = np.ones((15,15)) * np.nan
        T[0,[ 1, 2]] = 1
        T[1,[ 3, 4]] = 1
        T[2,[ 5, 6]] = 1
        T[3,[ 7, 8]] = 1
        T[4,[ 9,10]] = 1
        T[5,[11,12]] = 1
        T[6,[13,14]] = 1
        T[np.arange(7,15),np.arange(7,15)] = 1

        ## Define rewards.
        R = np.copy(T)
        R[np.where(~np.isnan(R))] = 0
        R[0,[ 1, 2]] = [-70,-20]
        R[1,[ 3, 4]] = [-20,-70]
        R[2,[ 5, 6]] = [-20,-70]
        R[3,[ 7, 8]] = [-20, 20]
        R[4,[ 9,10]] = [ 20,140]
        R[5,[11,12]] = [-20, 20]
        R[6,[13,14]] = [-20, 20]
        
        ## Define start/terminal states.
        start = 0
        terminal = np.arange(7,15)

        ## Initialize GraphWorld.
        GraphWorld.__init__(self, T, R, start, terminal, epsilon=0)
        
    def __repr__(self):
        return '<GraphWorld | Decision Tree>'
    
    def _draw_nodes(self, ax, xpos, ypos, s=1000, color=None, cmap=None, vmin=None, vmax=None, 
                alpha=1.0, linewidth=1.0):
        """Draw decision tree nodes. See plot_decision tree for details."""

        from matplotlib.cm import get_cmap
        from matplotlib.colors import ListedColormap, Normalize
        
        ## Define colors.
        if color is None: 
            colors = np.repeat('#1f77b4', len(xpos))
        elif isinstance(color, str):
            colors = np.repeat(color, len(xpos))
        elif np.issubdtype(np.array(color).dtype, np.number):
            assert np.equal(len(color), len(xpos))        
            if not isinstance(cmap, ListedColormap): cmap = get_cmap(cmap)
            colors = cmap(Normalize(vmin, vmax)(np.array(color)))
        else:
            assert np.equal(len(color), len(xpos))        
            colors = np.copy(color)

        ## Define transparency.
        if isinstance(alpha, float): 
            alphas = np.repeat(alpha, len(xpos))
        else: 
            assert np.equal(len(alpha), len(xpos))
            alphas = np.copy(alpha)

        ## Iteratively plot.
        for i, (x, y, color, alpha) in enumerate(zip(xpos, ypos, colors, alphas)):
            ax.scatter(x, y, s=s, color='w', alpha=1)
            ax.scatter(x, y, s=s, color=color, alpha=alpha, linewidth=linewidth, 
                       edgecolor='k')

        return ax

    def _draw_node_labels(self, ax, xpos, ypos, fontsize=14):
        """Draw decision tree node labels. See plot_decision tree for details."""

        for i, (x, y) in enumerate(zip(xpos, ypos)):
            ax.text(x, y, '%0.0f' %(i+1), ha='center', va='center', fontsize=fontsize)

        return ax

    def _draw_edges(self, ax, xpos, ypos, edges, linewidth=1, color='0.5'):
        """Draw decision tree edges. See plot_decision tree for details."""

        ## Define line widths.
        if isinstance(linewidth, (int, float)): 
            linewidth = np.repeat(linewidth, len(edges))

        ## Iteratively draw.
        for i, (s1, s2) in enumerate(edges):
            ax.plot([xpos[s1], xpos[s2]], [ypos[s1], ypos[s2]], color=color, 
                    lw=linewidth[i], zorder=0)

        return ax

    def _draw_edge_labels(self, ax, labels, fontsize=14, alpha=1):

        ## Define label positions.
        xpos = [-0.6,0.6,-1.3,-0.7,0.7,1.3,-1.67,-1.33,-0.67,-0.33,0.33,0.67,1.33,1.67]
        ypos = [2.5] * 2 + [1.5] * 4 + [0.5] * 8
        halign = ['right','left'] * 7

        ## Define label transparency.
        if isinstance(alpha, (int, float)): alpha = np.repeat(alpha, len(xpos))

        ## Draw edge labels.
        for x, y, r, ha, a in zip(xpos, ypos, labels, halign, alpha):
            ax.text(x,y,'%0.0f' %r, va='center', ha=ha, fontsize=fontsize, alpha=a)

        return ax

    def _draw_path_sums(self, ax, xpos, linewidth=5, fontsize=14, alpha=1.0):

        ## Define path sums.
        sums = [-110, -70, -120, 0, -60, -20, -110, -70]

        ## Define transparency.
        if isinstance(alpha, (int, float)): alpha = np.repeat(alpha, len(sums))
        alpha = alpha[-8:]

        ## Draw line.
        ax.hlines(-0.40, -2, 2, lw=linewidth, color='k')

        ## Draw text.
        for x, r, a in zip(xpos, sums, alpha):
            ax.text(x, -0.70, '%0.0f' %r, ha='center', va='center', fontsize=fontsize, alpha=a)

        return ax

    def plot_decision_tree(self, s=1000, color=None, cmap=None, vmin=None, vmax=None, alpha=1.0, 
                           node_width=1.0, node_labels=False, edge_width=1.0, edge_labels=False, 
                           edge_label_alpha=1.0, path_sums=True, ax=None):
        """Plot decision tree environment.
        
        Parameters
        ----------
        s : float
            Node size.
        color : str
            Node color.
        cmap : matplotlib colormap name or object, or list of colors
            The mapping from data values to color space.
        vmin, vmax : floats
            Values to anchor the colormap, otherwise they are inferred 
            from the data and other keyword arguments.
        alpha : float
            Node transparancy.
        node_width : float
            Width of node edges.
        node_labels : bool
            Draw node labels. 
        edge_width : float
            Width of edges.
        edge_labels : bool or list
            Draw edge labels.
        edge_label_alpha : float
            Transparancy of edges.
        path_sums : bool
            Draw path sums.
        ax : matplotlib Axes
            Axes in which to draw the plot.

        Returns
        -------
        ax : matplotlib Axes
            Axes in which to draw the plot.
        """
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        ## Initialize canvas.
        if ax is None: fig, ax = plt.subplots(1,1,figsize=(5,5))
        
        ## Define decision tree.
        T = np.zeros((15,15))
        T[0,[ 1, 2]] = [-70,-20]
        T[1,[ 3, 4]] = [-20,-70]
        T[2,[ 5, 6]] = [-20,-70]
        T[3,[ 7, 8]] = [-20, 20]
        T[4,[ 9,10]] = [ 20,140]
        T[5,[11,12]] = [-20, 20]
        T[6,[13,14]] = [-20, 20]

        ## Define decision tree edges.
        edges = [arr.squeeze() for arr in np.array([np.where(T)]).T]

        ## Define node positions.
        xpos = [0,-1,1,-1.5,-0.5,0.5,1.5,-1.75,-1.25,-0.75,-0.25,0.25,0.75,1.25,1.75]
        ypos = [3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

        ## Draw DAG.
        ax = self._draw_edges(ax, xpos, ypos, edges=edges, linewidth=edge_width)
        ax = self._draw_nodes(ax, xpos, ypos, s=s, color=color, cmap=cmap, vmin=vmin, vmax=vmax,
                            alpha=alpha, linewidth=node_width)

        ## Optional details.
        if path_sums: ax = self._draw_path_sums(ax, xpos[-8:], alpha=alpha)
        if node_labels: ax = self._draw_node_labels(ax, xpos, ypos)
        if isinstance(edge_labels, (list, tuple, np.ndarray)): 
            ax = self._draw_edge_labels(ax, edge_labels, alpha=edge_label_alpha)
        elif np.equal(edge_labels, True):
            ax = self._draw_edge_labels(ax, T[T.nonzero()], alpha=edge_label_alpha)

        ## Clean up.
        ax.set(xlim=(-2,2), xticks=[], yticks=[])
        sns.despine(top=True, right=True, bottom=True, left=True, ax=ax)

        return ax