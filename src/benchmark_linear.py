import numpy as np
import matplotlib.pyplot as plt
import satlas2
import time

np.random.seed(0)

a, b = 3, 1
sigma = 2

x = np.linspace(1, 20, 5)
# x = x - x.mean()
y = a * x + b + np.random.randn(x.shape[0]) * sigma * sigma
yerr = np.ones(y.shape) * sigma

f = satlas2.Fitter()
d = satlas2.Source(x, y, yerr, name='LinearTest')
m = satlas2.Polynomial([a, b], name='Linear')
d.addModel(m)
f.addSource(d)

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
ax.errorbar(x, y, yerr, fmt='.')
ax.plot(d.x, d.evaluate(d.x), label='Original')

# f.fit(llh_selected=True, method='slsqp')
# f.fit()
# start = time.time()
# f.fit(llh_selected=True, method='emcee', steps=1000, filename='linear.h5')
# stop = time.time()
# print(f.reportFit())
# print(stop - start)
ax.plot(d.x, d.evaluate(d.x), label='Fit')

ax.legend(loc=0)
satlas2.generateCorrelationPlot('benchmark.h5', selection=(10, 100), filter=['Al', 'Au', 'Bu', 'Centroid'])
satlas2.generateWalkPlot('benchmark.h5', filter=['Al', 'Au', 'Bu', 'Centroid'])

plt.show()