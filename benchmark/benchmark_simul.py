import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '..\src')

import satlas as sat

import satlas2


def modifiedSqrt(input):
    output = np.sqrt(input)
    output[input <= 0] = 1e-12
    return output


spin = 3.5
J = [0.5, 1.5]
A = [9600, 175]
B = [0, 315]
C = [0, 0]
FWHMG = 135
FWHML = 101
centroid = 480
bkg = 10
scale = 90

x = np.arange(-17500, -14500, 40)
x = np.hstack([x, np.arange(20000, 23000, 40)])
times = []

rng = np.random.default_rng(0)
for j in range(1, 4):
    f = satlas2.Fitter()
    for i in range(j):
        hfs = satlas2.HFS(spin,
                          J,
                          A=A,
                          B=B,
                          C=C,
                          scale=scale,
                          df=centroid,
                          name='HFS1',
                          racah=True,
                          fwhmg=135,
                          fwhml=100)
        bkgm = satlas2.Polynomial([bkg], name='bkg1')
        y = hfs.f(x) + bkgm.f(x)
        y = rng.poisson(y)

        datasource = satlas2.Source(x,
                                    y,
                                    yerr=modifiedSqrt,
                                    name='Scan{}'.format(i + 1))

        datasource.addModel(hfs)
        datasource.addModel(bkgm)
        f.addSource(datasource)
    f.shareModelParams(
        ['Al', 'Au', 'Bl', 'centroid', 'FWHMG', 'FWHML', 'scale', 'p0'])
    print('Fitting {} datasets with chisquare (Pearson)...'.format(j))
    start = time.time()
    f.fit()
    stop = time.time()
    dt = stop - start
    print('{:.3} s, {:.0f} function evaluations'.format(dt, f.result.nfev))
    times.append(dt)

f.revertFit()
fig = plt.figure(constrained_layout=True)
gs = gridspec.GridSpec(nrows=len(f.sources), ncols=2, figure=fig)
a1 = None
a2 = None
axes = []
for i, (name, datasource) in enumerate(f.sources):
    if a1 is None:
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        a1 = ax1
        a2 = ax2
    else:
        ax1 = fig.add_subplot(gs[i, 0], sharex=a1)
        ax2 = fig.add_subplot(gs[i, 1], sharex=a2)
    left = datasource.x < 0
    right = datasource.x > 0
    smooth_left = np.arange(datasource.x[left].min(), datasource.x[left].max(),
                            5.0)
    smooth_right = np.arange(datasource.x[right].min(),
                             datasource.x[right].max(), 5.0)
    ax1.plot(datasource.x[left],
             datasource.y[left],
             drawstyle='steps-mid',
             label='Data')
    ax1.plot(smooth_left, datasource.evaluate(smooth_left), label='Initial')
    ax2.plot(datasource.x[right],
             datasource.y[right],
             drawstyle='steps-mid',
             label='Data')
    ax2.plot(smooth_right, datasource.evaluate(smooth_right), label='Initial')
    ax1.set_xlabel('Frequency [MHz]')
    ax2.set_xlabel('Frequency [MHz]')
    ax1.set_ylabel('Counts')
    ax2.set_ylabel('Counts')
    ax1.label_outer()
    ax2.label_outer()
    axes.append([ax1, ax2])
f.fit()
# print(f.reportFit())
for i, (name, datasource) in enumerate(f.sources):
    smooth_left = np.arange(datasource.x[left].min(), datasource.x[left].max(),
                            5.0)
    smooth_right = np.arange(datasource.x[right].min(),
                             datasource.x[right].max(), 5.0)
    axes[i][0].plot(smooth_left, datasource.evaluate(smooth_left), label='Fit')
    axes[i][1].plot(smooth_right,
                    datasource.evaluate(smooth_right),
                    label='Fit')
a1.legend(loc=0)

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(range(1, len(times) + 1), times)
ax.set_xlabel('Number of datasets')
ax.set_ylabel('Fitting time in seconds')

plt.show()