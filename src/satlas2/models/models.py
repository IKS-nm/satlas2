from satlas2.core import Model, Parameter

import numpy as np
import uncertainties as unc
from scipy.special import wofz, erf
from sympy.physics.wigner import wigner_6j, wigner_3j

__all__ = ['ExponentialDecay', 'Polynomial', 'Voigt', 'SkewedVoigt']

sqrt2 = 2 ** 0.5
sqrt2log2t2 = 2 * np.sqrt(2 * np.log(2))
log2 = np.log(2)

class Polynomial(Model):
    def __init__(self, p, name=None, prefunc=None):
        super().__init__(name=name, prefunc=prefunc)
        self.params = {'p'+str(len(p)-(i+1)): Parameter(value=P, min=-np.inf, max=np.inf, vary=True) for i, P in enumerate(p)}

    def f(self, x) -> float:
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

class Voigt(Model):
    def __init__(self, A, mu, FWHMG, FWHML, name=None, prefunc=None):
        super().__init__(name=name, prefunc=prefunc)
        self.params = {'A': Parameter(value=A, min=0, max=np.inf, vary=True),
                       'mu': Parameter(value=mu, min=-np.inf, max=np.inf, vary=True),
                       'FWHMG': Parameter(value=FWHMG, min=0, max=np.inf, vary=True),
                       'FWHML': Parameter(value=FWHML, min=0, max=np.inf, vary=True)
                       }

    def f(self, x):
        x = self.transform(x)
        A = self.params['A'].value
        mu = self.params['mu'].value
        x = x-mu
        FWHMG = self.params['FWHMG'].value
        FWHML = self.params['FWHML'].value
        sigma, gamma = FWHMG / sqrt2log2t2, FWHML / 2
        z = (x + 1j * gamma) / (sigma * sqrt2)
        ret = wofz(z).real
        n = wofz(1j * FWHML / (FWHMG * sqrt2)).real
        return A * ret / n

    def calculateFWHM(self):
        G, Gu = self.params['FWHMG'].value, self.params['FWHMG'].unc
        L, Lu = self.params['FWHML'].value, self.params['FWHML'].unc
        correl = self.params['FWHMG'].correl['FWHML']
        G, L = unc.correlated_values_norm([(G, Gu), (L, Lu)], np.array([[1, correl], [correl, 1]]))
        fwhm = 0.5346*L+(0.2166*L*L+G*G)**0.5
        return fwhm.nominal_value, fwhm.std_dev

class SkewedVoigt(Voigt):
    def __init__(self, A, mu, FWHMG, FWHML, skew, name=None, prefunc=None):
        super().__init__(A, mu, FWHMG, FWHML, name=name, prefunc=prefunc)
        self.params['Skew'] = Parameter(value=skew, min=-np.inf, max=np.inf, vary=True)

    def f(self, x):
        ret = super().f(x)
        mu = self.params['mu'].value
        FWHMG = self.params['FWHMG'].value
        sigma = FWHMG / sqrt2log2t2
        skew = self.params['Skew'].value
        beta = skew / (sigma*sqrt2)
        asym = 1 + erf(beta*(x-mu))
        return ret * asym
