import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

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
bkg = 0
scale = 10

hfs = satlas2.HFS(spin, J, A, B, C, df=centroid, fwhmg=FWHMG, fwhml=FWHML, scale=scale)
background = satlas2.Polynomial([bkg])

models = [hfs, background]

x = np.arange(-400, 300, 30)
y = satlas2.generateSpectrum(models, x)
plt.plot(x, y, drawstyle='steps-mid', label='Binned generation')
import functools
f = lambda x: np.random.default_rng().normal(x, 0.5)
y = satlas2.generateSpectrum(models, x, generator=f)
plt.plot(x, y, drawstyle='steps-mid', label='Binned Gaussian')

x = np.arange(-400, 300)
plt.plot(x, hfs.f(x)+background.f(x), label='Distribution')
plt.legend(loc=0)

plt.show()
