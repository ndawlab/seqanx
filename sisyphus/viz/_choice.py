import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

def _draw_nodes(ax, s, free_color, fixed_color):
    
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

def _draw_edges(ax):

    ## Draw edges.
    ax.plot([0.0, 2.0], [0.0, 0.5], color='k', zorder=0)
    ax.plot([0.0, 2.0], [0.0,-0.5], color='k', zorder=0)
    ax.plot([1.0, 2.0], [0.25,0.0], color='k', zorder=0)

    return ax

def _draw_edge_labels(ax, fontsize=14):
    
    ax.text(0.5,  0.15, '0', ha='center', va='bottom', fontsize=fontsize)
    ax.text(0.5, -0.15, '0', ha='center', va='top', fontsize=fontsize)
    ax.text(1.5,  0.40, r'$R$', ha='center', va='bottom', fontsize=fontsize)
    ax.text(1.5,  0.10, r'$R$', ha='center', va='top', fontsize=fontsize)
    ax.text(1.5, -0.40, r'$R$', ha='center', va='top', fontsize=fontsize)
    ax.annotate(r'$R = \{0.33: -1, 0.33: 0, 0.33: 1\}$', (0,0), (0,0), 
                xycoords='axes fraction', ha='left', va='bottom', fontsize=fontsize-4)
    return ax

def plot_free_choice(ax, s=50, free_color='#834c7d', fixed_color='0.9',
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
    
    ## Draw DAG.
    ax = _draw_edges(ax)
    ax = _draw_nodes(ax, s=50, free_color='#834c7d', fixed_color='0.9')
    
    ## Optional details.
    if edge_labels: ax = _draw_edge_labels(ax, fontsize=18)
    
    ## Clean up.
    ax.set(xlim=(-0.5,2.5), xticks=[], ylim=(-0.7,0.7), yticks=[])
    sns.despine(top=True, right=True, bottom=True, left=True, ax=ax)   
    
    return ax