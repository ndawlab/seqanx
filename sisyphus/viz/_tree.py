import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, Normalize

def _draw_nodes(ax, xpos, ypos, s=1000, color=None, cmap=None, vmin=None, vmax=None, 
                alpha=1.0, linewidth=1.0):
    """Draw decision tree nodes. See plot_decision tree for details."""
    
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

def _draw_node_labels(ax, xpos, ypos, fontsize=14):
    """Draw decision tree node labels. See plot_decision tree for details."""
   
    for i, (x, y) in enumerate(zip(xpos, ypos)):
        ax.text(x, y, '%0.0f' %(i+1), ha='center', va='center', fontsize=fontsize)
        
    return ax

def _draw_edges(ax, xpos, ypos, edges, linewidth=1, color='0.5'):
    """Draw decision tree edges. See plot_decision tree for details."""
    
    ## Define line widths.
    if isinstance(linewidth, (int, float)): 
        linewidth = np.repeat(linewidth, len(edges))
        
    ## Iteratively draw.
    for i, (s1, s2) in enumerate(edges):
        ax.plot([xpos[s1], xpos[s2]], [ypos[s1], ypos[s2]], color=color, 
                lw=linewidth[i], zorder=0)

    return ax

def _draw_edge_labels(ax, labels, fontsize=14, alpha=1):
    
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
        
def _draw_path_sums(ax, xpos, linewidth=5, fontsize=14, alpha=1.0):
    
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
    
def plot_decision_tree(ax, s=1000, color=None, cmap=None, vmin=None, vmax=None, alpha=1.0, 
                       node_width=1.0, node_labels=False, edge_labels=False, edge_width=1.0, 
                       edge_label_alpha=1.0, path_sums=True):

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
    ax = _draw_edges(ax, xpos, ypos, edges=edges, linewidth=edge_width)
    ax = _draw_nodes(ax, xpos, ypos, s=s, color=color, cmap=cmap, vmin=vmin, vmax=vmax,
                        alpha=alpha, linewidth=node_width)
    
    ## Optional details.
    if path_sums: ax = _draw_path_sums(ax, xpos[-8:], alpha=alpha)
    if node_labels: ax = _draw_node_labels(ax, xpos, ypos)
    if isinstance(edge_labels, (list, tuple, np.ndarray)): 
        ax = _draw_edge_labels(ax, edge_labels, alpha=edge_label_alpha)
    elif np.equal(edge_labels, True):
        ax = _draw_edge_labels(ax, T[T.nonzero()], alpha=edge_label_alpha)
    
    ## Clean up.
    ax.set(xlim=(-2,2), xticks=[], yticks=[])
    sns.despine(top=True, right=True, bottom=True, left=True, ax=ax)
    
    return ax