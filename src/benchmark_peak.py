import numpy as np
import matplotlib.pyplot as plt
import satlas2

def modifiedSqrt(input):
    output = np.sqrt(input)
    output[input<=0] = 1e-12
    return output

x = np.linspace(-100, 100, 50)

f = satlas2.Fitter()
m = satlas2.HFS(0, [0.5, 1.5], scale=10, name='model', fwhm=20)
m2 = satlas2.Polynomial([1], name='bkg')
rng = np.random.default_rng(0)
y = m.f(x) + m2.f(x)
y = rng.poisson(y)
d = satlas2.Source(x, y, modifiedSqrt, name='Data')
d.addModel(m)
d.addModel(m2)
f.addSource(d)
# y = np.random.Generator.poisson(lam=y)
f.fit(llh_selected=True, method='nelder', llh_method='poisson')
# f.fit()
print(f.reportFit())
plot_x = np.linspace(-100, 100, 200)
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
ax.plot(x, y, drawstyle='steps')
# ax.errorbar(x, y, yerr, fmt='.')
ax.plot(plot_x, d.evaluate(plot_x))
ax.grid()
plt.show()