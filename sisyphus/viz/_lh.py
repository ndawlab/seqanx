import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import radians as rad
from matplotlib.patches import Arc, RegularPolygon

def _draw_nodes(ax, s):
    
    ## Define node positions.
    xpos = [-1, 0, 1, 1]
    ypos = [0.0, 0.0, 0.5,-0.5]
    
    ## Draw nodes.
    ax.scatter(xpos, ypos, s)
    
    return ax

def _draw_edges(ax):

    ## Draw directed edges.
    ax.plot([-1.0, 0.0], [0.0, 0.0], color='k', zorder=0)
    ax.plot([ 0.0, 1.0], [0.0, 0.5], color='k', zorder=0)
    ax.plot([ 0.0, 1.0], [0.0,-0.5], color='k', zorder=0)
    
    ## Draw self-loop.
    ax = _draw_self_loop(ax, 0.4, -1.25, 0, 70, -155)
    
    return ax

def _draw_self_loop(ax, radius, centX, centY, angle_, theta2_, color_='black'):
    
    ## Construct line.
    arc = Arc([centX,centY],radius,radius,angle=angle_,
          theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=1,color=color_)
    ax.add_patch(arc)

    ## Construct arrow head.
    endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
    endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))

    ax.add_patch(                    #Create triangle as arrow head
        RegularPolygon(
            (endX, endY),            # (x,y)
            3,                       # number of vertices
            radius/9,                # radius
            rad(angle_+theta2_),     # orientation
            color=color_
        )
    )
    
    return ax

def plot_helplessness(ax):
    
    _draw_edges(ax)
    _draw_nodes(ax, 2500)

    ## Clean up.
    ax.set(xticks=[], yticks=[], ylim=(-1.5, 1.5), xlim=(-1.6, 1.6))
    sns.despine(top=True, bottom=True, left=True, right=True, ax=ax)