import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '..\..\..\src')

import satlas2


def modifiedSqrt(input):
    output = np.sqrt(input)
    output[input <= 0] = 1
    return output


amplitude = 10
halflife = 3
mu = 4.5
FWHMG = 1.5
FWHML = 1.5
model1 = satlas2.ExponentialDecay(amplitude, halflife, name='Background')
model2 = satlas2.Voigt(amplitude, mu, FWHMG, FWHML, name='Signal')

rng = np.random.default_rng(0)

data_x = np.arange(0, 10, 0.5)
plot_x = np.linspace(data_x.min(), data_x.max(), 10 * len(data_x))

data_y = satlas2.generateSpectrum([model1, model2], data_x, rng.poisson)

datasource = satlas2.Source(data_x,
                            data_y,
                            yerr=modifiedSqrt,
                            name='ArtificialData1')
datasource.addModel(model1)
datasource.addModel(model2)
model1.params['amplitude'].min = 0
model1.params['halflife'].min = 0
model2.params['A'].min = 0
f = satlas2.Fitter()
f.addSource(datasource)
f.setExpr('ArtificialData1___Signal___FWHML',
          'ArtificialData1___Signal___FWHMG')

size = 1.5
fig = plt.figure(constrained_layout=True, figsize=(16/2*size, 9/2*size))
gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)
ax_base = fig.add_subplot(gs[:, 0])
ax_evaluatewalk = fig.add_subplot(gs[0, 1], sharex=ax_base, sharey=ax_base)
ax_banddata = fig.add_subplot(gs[1, 1], sharex=ax_base, sharey=ax_base)
ax_bandmean = fig.add_subplot(gs[2, 1], sharex=ax_base, sharey=ax_base)

ax_base.plot(datasource.x, datasource.y, drawstyle='steps-mid', label='Data')
ax_evaluatewalk.plot(datasource.x, datasource.y, drawstyle='steps-mid')
lcd, = ax_banddata.plot(datasource.x, datasource.y, drawstyle='steps-mid')
ax_bandmean.plot(datasource.x, datasource.y, drawstyle='steps-mid')

ax_base.plot(plot_x, datasource.evaluate(plot_x), label='True spectrum')
ax_evaluatewalk.plot(plot_x, datasource.evaluate(plot_x))
ax_banddata.plot(plot_x, datasource.evaluate(plot_x))
ax_bandmean.plot(plot_x, datasource.evaluate(plot_x))
ax_base.label_outer()
ax_evaluatewalk.label_outer()
ax_bandmean.label_outer()
ax_banddata.label_outer()

ax_base.set_ylabel('Counts')

filename = 'evaluateTest.h5'

try:
    f.readWalk(filename)
except Exception as e:
    f.fit(llh=True, llh_method='poisson')
    f.fit(llh=True, llh_method='poisson', method='emcee', filename=filename)
f.fit(llh=True, llh_method='poisson')
ax_base.plot(plot_x, datasource.evaluate(plot_x), label='Fit')
ax_evaluatewalk.plot(plot_x, datasource.evaluate(plot_x))
ax_banddata.plot(plot_x, datasource.evaluate(plot_x))
lcm, = ax_bandmean.plot(plot_x, datasource.evaluate(plot_x))

print(f.reportFit())
burnin = 100
X, band = f.evaluateOverWalk(filename, burnin=burnin, evals=5000, x=plot_x)
band = band[0]
lc, = ax_evaluatewalk.plot(plot_x, band[1], color='grey')
ax_evaluatewalk.fill_between(plot_x,
                             band[0],
                             band[2],
                             color=lc.get_color(),
                             alpha=0.3,
                             label='1-$\sigma$')

lowerd1, upperd1 = satlas2.poissonInterval(datasource.y, sigma=1)
lowerd2, upperd2 = satlas2.poissonInterval(datasource.y, sigma=2)
lowerd3, upperd3 = satlas2.poissonInterval(datasource.y, sigma=3)

ax_banddata.fill_between(datasource.x,
                         lowerd1,
                         upperd1,
                         color=lcd.get_color(),
                         alpha=0.3,
                         step='mid',
                         label='1-$\sigma$')
# ax_banddata.fill_between(datasource.x, lowerd2, upperd2, color=lcd.get_color(), alpha=0.3, step='mid')
# ax_banddata.fill_between(datasource.x, lowerd3, upperd3, color=lcd.get_color(), alpha=0.3, step='mid')

plot_y = datasource.evaluate(plot_x)
lowerm1, upperm1 = satlas2.poissonInterval(plot_y, sigma=1, mean=True)
lowerm2, upperm2 = satlas2.poissonInterval(plot_y, sigma=2, mean=True)
lowerm3, upperm3 = satlas2.poissonInterval(plot_y, sigma=3, mean=True)

ax_bandmean.fill_between(plot_x,
                         lowerm1,
                         upperm1,
                         color=lcm.get_color(),
                         alpha=0.3,
                         step='mid',
                         label='1-$\sigma$')
# ax_bandmean.fill_between(plot_x, lowerm2, upperm2, color=lcm.get_color(), alpha=0.3, step='mid')
# ax_bandmean.fill_between(plot_x, lowerm3, upperm3, color=lcm.get_color(), alpha=0.3, step='mid')

ax_base.legend(loc=0)
ax_evaluatewalk.legend(loc=0)
ax_bandmean.legend(loc=0)
ax_banddata.legend(loc=0)

ax_evaluatewalk.set_title('Interval on mean\nbased on fit')
ax_bandmean.set_title('Interval on data\nbased on mean')
ax_banddata.set_title('Interval on mean\nbased on data')

# satlas2.generateCorrelationPlot(filename, burnin=burnin)
# satlas2.generateWalkPlot(filename, burnin=burnin)
plt.show()