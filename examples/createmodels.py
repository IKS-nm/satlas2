import sys

sys.path.insert(0, '..\src')
import matplotlib.pyplot as plt
import numpy as np

import satlas2


class ExpSine(satlas2.Model):
    def __init__(self, A, lamda, omega, name='ExpSine', prefunc=None):
        super().__init__(name, prefunc=prefunc)
        self.params = {
            'amplitude': satlas2.Parameter(value=A,
                                           min=0,
                                           max=np.inf,
                                           vary=True),
            'lambda': satlas2.Parameter(value=lamda,
                                        min=0,
                                        max=np.inf,
                                        vary=True),
            'omega': satlas2.Parameter(value=omega,
                                       min=0,
                                       max=np.inf,
                                       vary=True)
        }

    def f(self, x):
        x = self.transform(x)
        a = self.params['amplitude'].value
        l = self.params['lambda'].value
        o = self.params['omega'].value
        return a * np.exp(-l * x) * np.sin(o * x)


amplitude = 7
lamda = 1.5
omega = 4
model = ExpSine(amplitude, lamda, omega, name='MyModel')

x = np.linspace(0, 4, 100)
y = model.f(x)

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(x, y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('My ExpSine Model')
print(model.params)

data_x = np.linspace(0, 4, 20)
data_y = model.f(data_x) + np.random.randn(data_x.shape[0]) * 0.5
yerr = np.ones(data_y.shape) * 0.5

datasource = satlas2.Source(data_x, data_y, yerr=yerr, name='Datafile1')
datasource.addModel(model)
f = satlas2.Fitter()
f.addSource(datasource)
f.fit()
print(f.reportFit())
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.errorbar(data_x, data_y, yerr=yerr, fmt='o', label='Data')
ax.plot(x, y, label='Initial guess')
ax.plot(x, model.f(x), label='Fit')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc=0)
plt.show()
