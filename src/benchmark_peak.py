import numpy as np
import matplotlib.pyplot as plt
import satlas2
import time
import emcee


def modifiedSqrt(input):
    output = np.sqrt(input)
    output[input <= 0] = 1e-12
    return output


x = np.linspace(-100, 100, 50)

f = satlas2.Fitter()
m = satlas2.HFS(0, [0.5, 1.5], scale=10, name='model', fwhm=20)
m2 = satlas2.Polynomial([1], name='bkg')
m.params['centroid'].min = -100
m.params['centroid'].max = 100
m.params['FWHMG'].max = 100
m.params['FWHML'].max = 100
rng = np.random.default_rng(0)
y = m.f(x) + m2.f(x)
# y = m2.f(x)
y = rng.poisson(y)
d = satlas2.Source(x, y, modifiedSqrt, name='Data')
d.addModel(m)
d.addModel(m2)
f.addSource(d)
# y = np.random.Generator.poisson(lam=y)
start = time.time()
f.fit(llh_selected=True, method='slsqp', llh_method='poisson', scale_covar=False)
stop = time.time()
f.fit(llh_selected=True,
      method='emcee',
      llh_method='poisson',
      filename='peak.h5',
      steps=1000,
      nwalkers=25)
# f.fit()
print(f.reportFit())
print(stop - start)
# def poissonLlh(self):
#         model_calcs = self.f()
#         returnvalue = self.temp_y * np.log(model_calcs) - model_calcs
#         returnvalue[model_calcs <= 0] = -1e20
#         return returnvalue

def log_prior(x):
    if x > 0:
        return 0.0
    else:
        return -np.inf

def log_likelihood(x, y):
    # model_calcs = np.ones(y.shape) * x
    returnvalue = y * np.log(x) - x
    # returnvalue[model_calcs <= 0] = -1e20
    return np.sum(returnvalue)

def log_prob(x, y):
    lp = log_prior(x)
    if not np.isfinite(x):
        return -np.inf
    return lp + log_likelihood(x, y)

# ndim, nwalkers = 1, 50
# p0 = np.random.rand(nwalkers, ndim)+5
# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[y])
# sampler.run_mcmc(p0, 1000)


plot_x = np.linspace(-100, 100, 200)

# fig = plt.figure()
# ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
# ax.hist(sampler.get_chain(flat=True), 100, histtype='step')
# ax.plot(x, y, drawstyle='steps')
# ax.errorbar(x, y, yerr, fmt='.')
# ax.plot(plot_x, np.ones(plot_x.shape) * sampler.get_chain(flat=True).mean())
# ax.grid()

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
# ax.hist(sampler.get_chain(flat=True), 100, histtype='step')
ax.plot(x, y, drawstyle='steps')
# ax.errorbar(x, y, yerr, fmt='.')
ax.plot(plot_x, d.evaluate(plot_x))
ax.grid()
satlas2.generateCorrelationPlot('peak.h5', selection=(20, 100))
satlas2.generateWalkPlot('peak.h5')

# fig, ax = plt.subplots(1, figsize=(10, 7), sharex=True)
# samples = sampler.get_chain()
# labels = ["m"]
# ax.plot(samples[:, :, 0], "k", alpha=0.3)
# ax.set_xlim(0, len(samples))
# ax.set_ylabel(labels[0])
# ax.yaxis.set_label_coords(-0.1, 0.5)

# ax.set_xlabel("step number");

plt.show()