import numpy as np
import matplotlib.pyplot as plt
import satlas2
np.random.seed(0)

a, b = 5, 1
sigma = 0.5

x = np.linspace(1, 20, 3)
x = x - x.mean()
y = a * x + b + np.random.randn(x.shape[0]) * sigma
yerr = np.ones(y.shape) * sigma

f = satlas2.Fitter()
d = satlas2.Source(x, y, yerr, name='LinearTest')
m = satlas2.Polynomial([a, b], name='Linear')
d.addModel(m)
f.addSource(d)
f.fit(llh_selected=True, method='slsqp')
print(f.reportFit())

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
ax.errorbar(x, y, yerr, fmt='.')
ax.plot(d.x, d.evaluate(d.x))
plt.show()