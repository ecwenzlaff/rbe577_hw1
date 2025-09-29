# RBE 577 - Machine Learning for Robotics
# HW 1: Control Allocation via Deep Neural Networks
# ecwenzlaff@wpi.edu

from typing import Tuple, Union, List # specified types (esp. those using "Any") primarily convey intention and are not guaranteed to work with static type checking libraries 
import numpy as np
import matplotlib.pyplot as plt
import torch

class LineStructure:
    def __init__(self, 
                 x: Union[List, np.ndarray, torch.Tensor], 
                 y: Union[List, np.ndarray, torch.Tensor], 
                 z: Union[List, np.ndarray, torch.Tensor] = None, 
                 label: str = None, 
                 marker: str = 'none',
                 fillstyle: str = 'none',
                 linestyle: str = '-',
                 linewidth: float = 1.0,
                 color: Tuple[float, float, float] = None,
                 markerfacecolor: Tuple[float, float, float] = None):
        self.x, self.y, self.z, self.label, self.marker, self.fillstyle, self.linestyle, self.linewidth, self.color = x, y, z, label, marker, fillstyle, linestyle, linewidth, color
        if (markerfacecolor is None):
            self.markerfacecolor = self.color

def plotLineStructures(lslist: List[LineStructure], 
                       splitview: Tuple[int, int] = (1,1),
                       supertitle: str = None, 
                       xlabels: List[str] = None,   # should have the same length as lslist
                       ylabels: List[str] = None,   # should have the same length as lslist
                       zlabels: List[str] = None,   # should have the same length as lslist (if the number of 3D subplots needed < len(lslist), then the indices should be filled with some sort of placeholder like "None")
                       subtitles: List[str] = None, # should have the same length as lslist
                       enablegrid: bool = True, 
                       fig: plt.Figure = None, 
                       ax: plt.Axes = None) -> Tuple[plt.Figure, plt.Axes, List[plt.Line2D]]:
    # Before doing anything, figure out what type of subplots to do:
    sptuples = []
    subplotnums = np.zeros((len(lslist),))
    if (max(splitview) > 1):
        subplotnums = np.arange(0, len(lslist))
    for i in range(0, len(lslist)):
        sptuples.append((subplotnums[i], lslist[i]))
    projectview = None
    if any([(not (ls.z is None)) for ls in lslist]):
        projectview='3d'
    if ((fig is None) or (ax is None)):
        # Need to avoid using "add_subplot()" method in the loop below to avoid overwriting passed in plt.Axes objects.
        # Also, need to use 'subplot_kw' since "projection=" syntax only applies to "add_subplot()" method:
        fig, ax = plt.subplots(max(splitview[0],1), max(splitview[1],1), layout="tight", subplot_kw={"projection": projectview}) 
    linehandles = []
    # Add supertitle if one is specified:
    if (supertitle != None):
        fig.suptitle(supertitle, fontweight='bold')
    anylabels = False
    for ax_idx, curr_ls in sptuples:
        threeDplot = False
        anylabels = (anylabels or (curr_ls.label != None))
        if (curr_ls.z is not None): 
            # since curr_ls.z will either be "None" or an np.ndarray, "!=" won't work since 
            # that operator checks on value instead of identity
             threeDplot = True
        if (max(splitview) > 1):
            axref = ax.reshape(-1)[int(ax_idx)] # plt.Axes objects are only subscriptable if multiple columns and/or rows are specified at subplot initilialization
        else:
            axref = ax
        # Plot the data for the current line structure:
        if (threeDplot):
            curr_line, = axref.plot(curr_ls.x, curr_ls.y, curr_ls.z, 
                                 label=curr_ls.label, 
                                 marker=curr_ls.marker, 
                                 fillstyle=curr_ls.fillstyle, 
                                 linestyle=curr_ls.linestyle, 
                                 linewidth=curr_ls.linewidth,
                                 color=curr_ls.color,
                                 markerfacecolor=curr_ls.markerfacecolor)
        else:
            curr_line, = axref.plot(curr_ls.x, curr_ls.y, 
                                 label=curr_ls.label, 
                                 marker=curr_ls.marker, 
                                 fillstyle=curr_ls.fillstyle, 
                                 linestyle=curr_ls.linestyle, 
                                 linewidth=curr_ls.linewidth,
                                 color=curr_ls.color, 
                                 markerfacecolor=curr_ls.markerfacecolor)
        linehandles.append(curr_line)
        # Format the current axis based on line structure contents and input arguments:
        if (xlabels != None): 
            axref.set_xlabel(xlabels[int(ax_idx)], fontweight='bold')
        if (ylabels != None): 
            axref.set_ylabel(ylabels[int(ax_idx)], fontweight='bold')
        if ((zlabels != None) and (threeDplot)):
            axref.set_zlabel(zlabels[int(ax_idx)], fontweight='bold')
        if (subtitles != None): 
            axref.set_title(subtitles[int(ax_idx)])
        if (anylabels):
            axref.legend() # should capture all LineStructures ever written to the given 'ax' 
        axref.grid(enablegrid)
    return fig, ax, linehandles

if __name__ == '__main__':
    line3d = LineStructure(x=np.arange(0,10), y=torch.rand((10,)), z=3*torch.rand((10,)), label='Random 3D', color='r')
    line2d_1 = LineStructure(x=np.arange(0,100), y=100*torch.rand((100,)), label='Random 2D #1')
    line2d_2 = LineStructure(x=np.arange(0,100), y=100*torch.rand((100,)), label='Random 2D #2')
    fig3d, ax3d = plotLineStructures([line3d], supertitle="3D Random Test")[0:2]
    fig2d, ax2d = plotLineStructures([line2d_1], supertitle="2D Random Test")[0:2]
    fig2d, ax2d = plotLineStructures([line2d_2], fig=fig2d, ax=ax2d)[0:2]
    plt.show()