import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import functools

sys.path.insert(0, '..\src')

import satlas2

A = 10
mu = 25
FWHMG = 120
FWHML = 80
skew = 3

skewR = satlas2.SkewedVoigt(A = A,
                            mu = -mu,
                            FWHMG = FWHMG,
                            FWHML = FWHML,
                            skew = skew,
                            name = "RSkewedVoigt",
                            )

skewL = satlas2.SkewedVoigt(A = A,
                            mu = mu,
                            FWHMG = FWHMG,
                            FWHML = FWHML,
                            skew = -skew,
                            name = "LSkewedVoigt",
                            )

dummy_x = np.linspace(-400,400,801)
plt.plot(dummy_x, skewL.f(dummy_x), 'orange')
plt.plot(dummy_x, skewR.f(dummy_x), 'green')
plt.show()