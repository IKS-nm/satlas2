import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import functools

sys.path.insert(0, "..\src")

import satlas2


class CustomLlhFitter(satlas2.Fitter):
    def customLlh(self):
        try:
            bunches = self.bunches
        except:
            bunches = self.getSourceAttr("bunches")
            self.bunches = bunches
        try:
            data_counts = self.data_counts
        except:
            data_counts = self.temp_y * bunches
            self.data_counts = data_counts
        model_rates = self.f()
        model_counts = model_rates * bunches
        returnvalue = data_counts * np.log(model_counts) - model_counts
        returnvalue[model_counts <= 0] = -np.inf
        priors = self.gaussianPriorResid()
        if len(priors) > 1:
            priors = -0.5 * priors * priors
            returnvalue = np.append(returnvalue, priors)
        return returnvalue


def modifiedSqrt(y, bunches=None):
    yerr = np.sqrt(y * bunches) / bunches
    yerr[y <= 0] = 1 / bunches[y <= 0]
    return yerr


spin = 1
J = [0, 1]
A = [0, 50]
B = [0, 0]
C = [0, 0]

FWHMG = 135/4
FWHML = 101/4

centroid = 0
bkg = 0.0005
scale = 0.001

hfs = satlas2.HFS(spin, J, A, B, C, df=centroid, fwhmg=FWHMG, fwhml=FWHML, scale=scale, peak = 'skewvoigt', peak_kwargs = {'skew':{'value': 11, 'min':1}})

background = satlas2.Polynomial([bkg])

bunches_background = 100000
noise = min(10, bunches_background / 5)
# bunches_amplitude = 100000

# sampling_peak = satlas2.Voigt(bunches_amplitude, centroid, FWHMG*0.5, FWHML*0.5)
sampling_background = satlas2.Polynomial([bunches_background])

models = [hfs, background]

x = np.arange(-150, 150, 5)
y = []
rng = np.random.default_rng(10)
bunches = satlas2.generateSpectrum(
    sampling_background, x, lambda x: rng.normal(x, noise)
)
bunches = np.ones(bunches.shape) * np.max(bunches)
bunches = bunches.astype(int)
for X, evaluated in zip(x, bunches):
    Y = satlas2.generateSpectrum(
        models, np.array([X] * evaluated), rng.poisson
    )
    y.append(Y.sum() / evaluated)
y = np.array(y)

yerrCalc = functools.partial(modifiedSqrt, bunches=bunches)

# source = satlas2.Source(x, y, yerr=yerrCalc, name='Data', bunches=bunches)
spin = 1
J = [0, 1]
A = [0, 51]
B = [0, 0]
C = [0, 0]
FWHMG = 120/4
FWHML = 120/3
centroid = 9
bkg = 0.0003
scale = 0.001
hfs = satlas2.HFS(spin, J, A, B, C, df=centroid, fwhmg=FWHMG, fwhml=FWHML, scale=scale, peak = 'skewvoigt', peak_kwargs = {'skew':{'value': 11, 'min':1}})
hfs.params['Bu'].vary = False

source = satlas2.Source(x, y, yerr=modifiedSqrt(y, bunches), name='Data', bunches=bunches)

f = CustomLlhFitter()
f.addSource(source)
source.addModel(hfs)
source.addModel(background)
# f.fit(llh=True, llh_method='poisson')
# print(f.reportFit())
# print(f.reportFit())
# print(f.getSourceAttr('bunches'))

plot_x = np.arange(x.min(), x.max() + 1, 1)

size = 1.5
fig = plt.figure(
    constrained_layout=True, figsize=(16 / 2 * size, 9 / 2 * size)
)
gs = gridspec.GridSpec(nrows=3, ncols=1, figure=fig)
ax_events = fig.add_subplot(gs[0, :])
ax_counts = fig.add_subplot(gs[1, :])
ax_rate = fig.add_subplot(gs[2, :])

ax_events.plot(source.x, source.bunches, drawstyle="steps-mid")
ax_counts.plot(x, y * bunches, drawstyle="steps-mid", label="Data")
ax_rate.errorbar(x, y, yerr=source.yerr(), drawstyle="steps-mid", label="Data")
plot_y = hfs.f(plot_x) + background.f(plot_x)
ax_rate.plot(plot_x, plot_y, label="Initial")
f.fit()
plot_y = hfs.f(plot_x)+background.f(plot_x)
ax_rate.plot(plot_x, plot_y, label='Gaussian fit')
print(f.reportFit())
f.revertFit()
f.fit(llh=True, llh_method="custom")
print(f.reportFit())
plot_y = hfs.f(plot_x)+background.f(plot_x)
ax_rate.plot(plot_x, plot_y, label='Likelihood fit')
f.revertFit()
f.fit(llh=True, method='emcee', llh_method='custom', filename = 'extraattributes.h5', steps = 2000)
print(f.reportFit())
plot_y = hfs.f(plot_x)+background.f(plot_x)
ax_rate.plot(plot_x, plot_y, label='Walker fit')


ax_rate.legend(loc=0)
ax_events.label_outer()
ax_counts.label_outer()
ax_rate.label_outer()

ax_rate.set_ylabel("Rate [cts/bunch]")
ax_events.set_ylabel("Number of bunches [-]")
ax_counts.set_ylabel("Raw counts")

data = np.vstack([x, y*bunches]).T
np.savetxt('extraattributesdata.txt', data, delimiter=',')
plt.show()

fig,ax = satlas2.generateWalkPlot(filename='extraattributes.h5')
plt.show()


fig,ax,cbar = satlas2.generateCorrelationPlot(filename='extraattributes.h5', burnin = 500)
plt.show()
