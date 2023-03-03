import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '..\src')

import satlas2


def modifiedSqrt(input):
    output = np.sqrt(input)
    output[input <= 0] = 1e-12
    return output


amplitude = 6
mu = 4.5
FWHMG = 2
FWHML = 1
model1 = satlas2.Polynomial([2, 3], name='Line1')
model2 = satlas2.Polynomial([6], name='Line2')
model3 = satlas2.Voigt(amplitude, mu, FWHMG, FWHML, name='Voigt')

rng = np.random.default_rng(0)

data_x1 = np.linspace(0, 5, 2)
data_x2 = np.linspace(1, 10, 11)

scale1 = 10
scale2 = 0.5

noise1 = rng.normal(loc=0, scale=scale1, size=data_x1.shape[0])
noise2 = rng.normal(loc=0, scale=scale2, size=data_x2.shape[0])

data_y1 = model1.f(data_x1) + noise1
yerr1 = np.ones(data_y1.shape) * scale1
data_y2 = model2.f(data_x2) + model3.f(data_x2) + noise2
yerr2 = np.ones(data_y2.shape) * scale2

datasource1 = satlas2.Source(data_x1,
                             data_y1,
                             yerr=yerr1,
                             name='ArtificialData1')
datasource2 = satlas2.Source(data_x2,
                             data_y2,
                             yerr=yerr2,
                             name='ArtificialData2')
datasource1.addModel(model1)
datasource2.addModel(model2)
datasource2.addModel(model3)
f = satlas2.Fitter()
f.addSource(datasource1)
f.addSource(datasource2)

fig = plt.figure(constrained_layout=True)
gs = gridspec.GridSpec(nrows=len(f.sources), ncols=1, figure=fig)
a1 = None
axes = []
for i, (name, datasource) in enumerate(f.sources):
    if a1 is None:
        ax1 = fig.add_subplot(gs[i, 0])
        a1 = ax1
    else:
        ax1 = fig.add_subplot(gs[i, 0], sharex=a1)
    ax1.errorbar(datasource.x,
                 datasource.y,
                 datasource.yerr(),
                 fmt='o',
                 label='Data')
    ax1.plot(datasource.x, datasource.f(), label='Initial')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.label_outer()
    axes.append(ax1)

filename = 'evaluateTest.h5'

try:
    f.readWalk(filename)
except Exception as e:
    f.fit()
    f.fit(method='emcee', filename=filename)
for i, (name, datasource) in enumerate(f.sources):
    ax1 = axes[i]
    ax1.plot(datasource.x, datasource.f(), label='Fit')
x = np.linspace(0, 10, 100)
print(f.reportFit())
X, bands = f.evaluateOverWalk(filename, burnin=100, evals=2000, x=x)

for i, (x, band) in enumerate(zip(X, bands)):
    ax = axes[i]
    lc, = ax.plot(x,
                  band[1],
                  label='Fit, with 1-sigma confidence bound',
                  color='grey')
    ax.fill_between(x, band[0], band[2], color=lc.get_color(), alpha=0.3)

a1.legend(loc=0)
satlas2.generateCorrelationPlot(filename)
plt.show()