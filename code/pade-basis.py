from phd import *
from utils import HandlerDashedLines, LineCollection
from nengolib.networks.rolling_window import _pade_readout as delay_readout

# Adapted from https://github.com/arvoelke/delay2017/blob/master/dodo.py

import pylab
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.serif'] = 'cm'

q = 12
theta = 1.0

thetas = np.linspace(0, theta, 1000)
data = np.zeros((q, len(thetas)))

for i, thetap in enumerate(thetas):
    data[:, i] = delay_readout(q, thetap / theta)

cmap = sns.color_palette("GnBu_d", len(data))  # sns.cubehelix_palette(len(data), light=0.7, reverse=True)

with sns.axes_style('ticks'):
    with sns.plotting_context('paper', font_scale=2.8):
        pylab.figure(figsize=(18, 5))
        for i in range(len(data)):
            pylab.plot(thetas / theta, data[i], c=cmap[i],
                       lw=3, alpha=0.7)

        pylab.xlabel(r"$\theta' \times \theta^{-1}$ [s / s]", labelpad=20)
        pylab.ylabel(r"$w_i$")
        lc = LineCollection(len(cmap) * [[(0, 0)]], lw=10,
                            colors=cmap)
        pylab.legend([lc], [r"$i = 0 \ldots q - 1$"], handlelength=2,
                     handler_map={type(lc): HandlerDashedLines()},
                     bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

        sns.despine(offset=15)

        savefig("pade-basis.pdf")
