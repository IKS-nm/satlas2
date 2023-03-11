import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, norm

sys.path.insert(0, '..\src')

import satlas2

spin = 0.5
J = [0.5, 1.5]
A = [100, 175]
B = [0, 0]
C = [0, 0]
FWHMG = 135/4
FWHML = 101/25
centroid = 0
bkg = 6
scale = 10

hfs = satlas2.HFS(spin, J, A, B, C, df=centroid, fwhmg=FWHMG, fwhml=FWHML, scale=scale)
background = satlas2.Polynomial([bkg])

models = [hfs, background]

x = np.arange(-400, 300, 30)
y = satlas2.generateSpectrum(models, x)

lowerd1, upperd1 = satlas2.poissonInterval(y, sigma=1)
lowerd2, upperd2 = satlas2.poissonInterval(y, sigma=2)
lowerd3, upperd3 = satlas2.poissonInterval(y, sigma=3)

plot_x = np.arange(-400, 300)
plot_y = hfs.f(plot_x)+background.f(plot_x)
lowerm1, upperm1 = satlas2.poissonInterval(plot_y, sigma=1, mean=True)
lowerm2, upperm2 = satlas2.poissonInterval(plot_y, sigma=2, mean=True)
lowerm3, upperm3 = satlas2.poissonInterval(plot_y, sigma=3, mean=True)

size = 1.5
fig = plt.figure(constrained_layout=True, figsize=(16/2*size, 9/2*size))
gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
ax_base = fig.add_subplot(gs[:, 0])
ax_intervaldata = fig.add_subplot(gs[0, 1], sharex=ax_base, sharey=ax_base)
ax_intervalspectrum = fig.add_subplot(gs[1, 1], sharex=ax_base, sharey=ax_base)

ax_base.plot(x, y, drawstyle='steps-mid', label='Data')
lc, = ax_intervaldata.plot(x, y, drawstyle='steps-mid', label='Data')
ax_intervalspectrum.plot(x, y, drawstyle='steps-mid', label='Data')
ax_intervaldata.fill_between(x, lowerd1, upperd1, color=lc.get_color(), alpha=0.3, step='mid', label='1, 2, 3-$\sigma$')
ax_intervaldata.fill_between(x, lowerd2, upperd2, color=lc.get_color(), alpha=0.3, step='mid')
ax_intervaldata.fill_between(x, lowerd3, upperd3, color=lc.get_color(), alpha=0.3, step='mid')

ax_base.plot(plot_x, plot_y, label='Spectrum')
ax_intervaldata.plot(plot_x, plot_y, label='Spectrum')
lc, = ax_intervalspectrum.plot(plot_x, plot_y, label='Spectrum')
ax_intervalspectrum.fill_between(plot_x, lowerm1, upperm1, color=lc.get_color(), alpha=0.3, step='mid', label='1, 2, 3-$\sigma$')
ax_intervalspectrum.fill_between(plot_x, lowerm2, upperm2, color=lc.get_color(), alpha=0.3, step='mid')
ax_intervalspectrum.fill_between(plot_x, lowerm3, upperm3, color=lc.get_color(), alpha=0.3, step='mid')

ax_base.legend(loc=0)
ax_intervaldata.legend(loc=0)
ax_intervalspectrum.legend(loc=0)

ax_base.label_outer()
ax_intervaldata.label_outer()
ax_intervalspectrum.label_outer()

ax_intervaldata.set_title('Confidence interval of mean\nbased on data')
ax_intervalspectrum.set_title('Confidence interval of data\nbased on mean')

plt.show()
