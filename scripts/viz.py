import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .agents._routines import softmax

def CanvasPolicy():
    
    ## Initialize canvas.
    fig = plt.figure(figsize=(21,5))

    ## Define boundaries of environment heatmap.
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.05, right=0.24, hspace=0)
    axes = [plt.subplot(gs[0])]

    ## Define boundaries of policy heatmaps.
    gs = gridspec.GridSpec(1, 3)
    gs.update(left=0.25, right=0.95, hspace=0)
    axes += [plt.subplot(gs[i]) for i in range(3)]
    
    return fig, axes

def plot_policy(env, Q, visits=None, beta=None, n_steps=100, color='0.6', cmap=None, 
                center=None, vmin=None, vmax=None, cbar=True, ax=None, cbar_ax=None):
        
    ## Prepare Q-values for visualization.
    if beta is None: V = Q.max(axis=1)
    else: V = np.sum(np.apply_along_axis(softmax, 1, Q) * Q, axis=1)
    V[env.terminal] = np.nan
    V = V.reshape(env.shape)
    
    ## Define boundaries of colobar.
    if ax is not None and cbar_ax is None:
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.01)
    
    ## Plot heatmap.
    ax = sns.heatmap(V, cmap=cmap, center=center, vmin=vmin, vmax=vmax, cbar=cbar,
                     xticklabels=[], yticklabels=[], ax=ax, cbar_ax=cbar_ax)
    
    ## Plot behavior.
    if visits is not None:
        
        ## Compute state occupancies. 
        states, counts = np.unique(visits, return_counts=True)
        
        ## Locate positions on plot.
        y, x = np.array([np.where(env.states == s) for s in states]).squeeze().T + 0.5
        sizes = 50 * counts / counts.max()
        ax.scatter(x,y,s=sizes,c=color)