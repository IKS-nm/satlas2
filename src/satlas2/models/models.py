from satlas2.core import Model, Parameter

import numpy as np
from scipy.special import wofz
from sympy.physics.wigner import wigner_6j, wigner_3j

__all__ = ['ExponentialDecay', 'Polynomial']

sqrt2 = 2 ** 0.5
sqrt2log2t2 = 2 * np.sqrt(2 * np.log(2))
log2 = np.log(2)

class Polynomial(Model):
    def __init__(self, p, name=None, prefunc=None):
        super().__init__(name=name, prefunc=prefunc)
        self.params = {'p'+str(len(p)-(i+1)): Parameter(value=P, min=-np.inf, max=np.inf, vary=True) for i, P in enumerate(p)}

    def f(self, x):
        x = self.transform(x)
        p = [self.params[paramkey].value for paramkey in self.params.keys()]
        return np.polyval(p, x)

class ExponentialDecay(Model):
    def __init__(self, a, tau, bkg, name=None, prefunc=None):
        super().__init__(name=name, prefunc=prefunc)
        self.params = {
            'amplitude': Parameter(value=a, min=-np.inf, max=np.inf, vary=True),
            'halflife': Parameter(value=tau, min=-np.inf, max=np.inf, vary=True),
            'background': Parameter(value=bkg, min=-np.inf, max=np.inf, vary=True),
        }

    def f(self, x):
        x = self.transform(x)
        a = self.params['amplitude'].value
        b = self.params['halflife'].value
        c = self.params['background'].value
        return a*np.exp(-log2*x/b)+c
