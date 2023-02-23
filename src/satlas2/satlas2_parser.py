"""
Implementation of the base HFSModel and SumModel classes, based on the syntax used in the original satlas

NOTE: THIS IS NOT FULLY BENCHMARKED/DEVELOPED SO BUGS MIGHT BE PRESENT, AND NOT ALL FUNCTIONALITIES OF THE ORIGINAL SATLAS ARE IMPLEMENTED

.. moduleauthor:: Bram van den Borne <bram.vandenborne@kuleuven.be>
.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""

import numpy as np
from .models import HFS, Polynomial, Step
from .core import Fitter, Source

class HFSModel:
    '''Initializes a hyperfine spectrum Model with the given hyperfine parameters.

    Parameters
    ----------
    I : float
        Integer or half-integer value of the nuclear spin
    J : ArrayLike
        A sequence of 2 spins, respectively the J value of the lower state
        and the J value of the higher state
    ABC : ArrayLike
        A sequence of 2 A, 2 B and 2 C values, respectively for the lower and the higher state
    centroid : float, optional
        The centroid of the spectrum, by default 0
    fwhm : ArrayLike, length = 2, optional
        First element: The Gaussian FWHM of the Voigt profile, by default 50
        Second element: The Lorentzian FWHM of the Voigt profile, by default 50
    scale : float, optional
        The amplitude of the entire spectrum, by default 1.0
    background_params: ArrayLike, optional
        The coefficients of the polynomial background, by default [0.001]
    shape : str, optional
        Voigt only
    use_racah : bool, optional
        Use individual amplitudes are setting the Racah intensities, by default True
    use_saturation : bool, optional
        False only
    saturation: float, optional
        No saturation
    sidepeak_params : dict, optional
        keys:
        N : int, optional
            Number of sidepeaks to be generated, by default None
        Poisson : float, optional
            The poisson factor for the sidepeaks, by default 0   
        Offset : float, optional
            Offset in units of x for the sidepeak, by default 0
    prefunc : callable, optional
        Transformation to be applied on the input before evaluation, by default None
'''
    def __init__(self,  I, J, ABC, centroid = 0, fwhm=[50.0,50.0], scale=1.0, background_params=[0.001], shape='voigt', use_racah=True, use_saturation=False, 
        saturation=0.001, sidepeak_params={'N': None, 'Poisson': 0, 'Offset': 0}, crystalballparams=None, 
        pseudovoigtparams=None, asymmetryparams=None, name = 'HFModel__'):
        super(HFSModel, self).__init__()
        self.background_params = background_params
        if shape != 'voigt':
            print('Only voigt shape is supported. Use satlas 2')
            raise NotImplementedError()
        if crystalballparams != None or pseudovoigtparams != None or asymmetryparams != None:
            print('Crystalball/Pseudovoigt/Asymmetric profiles not implemented. Use satlas 2')
            raise NotImplementedError()
        if name == 'HFModel__':
            self.name = name + str(I).replace('.', '_')
        self.hfs = HFS(I,
                              J,
                              A=ABC[:2],
                              B=ABC[2:4],
                              C=ABC[4:6],
                              scale=scale,
                              df=centroid,
                              fwhmg = fwhm[0],
                              fwhml = fwhm[1],
                              name=self.name.replace('.', '_'),
                              racah=use_racah,
                              N = sidepeak_params['N'],
                              offset = sidepeak_params['Offset'],
                              poisson = sidepeak_params['Poisson'],
                              prefunc = None)
        self.params = self.hfs.params

    def set_expr(self, constraints):
        """Set the expression to be used for the given parameters.
        The constraint should be a dict with following structure:
            key: string
                Parameter to constrain
            value: ArrayLike, length = 2
                First element: Factor to multiply
                Second element: Parameter that the key should be constrained to
            i.e.
            {'Au':['0.5','Al']} then Au = 0.5*Al"""
        for cons in constraints.keys():
            self.hfs.params[cons].expr = f'{constraints[cons][0]}*Fit___{self.name}___{constraints[cons][1]}'

    def fix_ratio(self, value, target = 'upper', parameter = 'A'):
        raise NotImplementedError('Use HFSModel.set_expr(...)')

    def f(self, x): 
        """Calculate the response for an unshifted spectrum with no background

        Parameters
        ----------
        x : ArrayLike

        Returns
        -------
        ArrayLike
        """
        return self.hfs.fUnshifted(x) 

    def __call__(self, x):
        """Calculate the response for an unshifted spectrum with background

        Parameters
        ----------
        x : ArrayLike

        Returns
        -------
        ArrayLike
        """
        return self.hfs.fUnshifted(x) + Polynomial(self.background_params, name='bkg').f(x)

    def chisquare_fit(self, x, y, yerr = None, xerr = None, func = None, verbose = None, hessian = False, method = 'leastsq', show_correl = True):
        """Perform a fit of this model to the data provided in this function.

        Parameters
        ----------
        x : ArrayLike
            x-values of the data points
        y : ArrayLike
            y-values of the data points
        yerr : ArrayLike
            1-sigma error on the y-values
        xerr : ArrayLike, optional
            1-sigma error on the x-values
        func: function, optional
            Not implemented
        verbose : Bool, optional
            Not implemented
        hessian : bool, optional
            Not implemented
        method : str, optional
            Selects the method used by the :func:`lmfit.minimizer`, by default 'leastsq'.
        show_correl : bool, optional
            `show correlations between fitted parameters in fit message, by default True

        Returns
        -------
        Instance of Fitter (from at satlas2)
        """
        if (func,verbose,hessian) != (None,None,False):
            raise NotImplementedError('Not implemented')
        datasource = Source(x,
                            y,
                            yerr=yerr,
                            name='Fit')
        datasource.addModel(self.hfs)
        bkg = Polynomial(self.background_params, name='bkg')
        datasource.addModel(bkg)
        f = Fitter()
        f.addSource(datasource)
        f.fit(method = method)
        self.background_params = [list(bkg.params.values())[i].value for i in range(len(list(bkg.params.values())))]
        print(f.reportFit(show_correl = show_correl))
        return f

class SumModel:
    """Initializes a hyperfine spectrum for the sum of multiple Models with the given models and a step background.

    Parameters
    ----------
    models : ArrayLike, with instances of HFSModel as elements
        The models that should be summed
    background_params: Dict with keys: 'values' and 'bounds' and values ArrayLike
        The bounds where the background changes stepwise in key 'bounds'
        The background values between the bounds
        i.e. {'values': [2,5], 'bounds':[-10]} means a background of 2 from -inf to -10, and a background of 5 from -10 to +inf
    name : string, optional
        Name of this summodel
    source_name : string, optional
        Name of the DataSource instance (from satlas2)
        """

    def __init__(self, models, background_params, name = 'sum', source_name='source'): 
        super(SumModel, self).__init__()
        self.name = name
        self.models = models
        self.background_params = background_params
        self._set_params()

    def _set_params(self): 
        """Set the parameters of the underlying Models
        based on a large Parameters object
        """
        for model in self.models:
            try:
                p.add_many(*model.params.values())
            except:
                p = model.params.copy()
        self.params = p

    def f(self, x):
        """Calculate the response for a spectrum

        Parameters
        ----------
        x : ArrayLike

        Returns
        -------
        ArrayLike
        """
        for model in self.models:
            try:
                f += model.f(x)
            except UnboundLocalError:
                f = model.f(x)
        return f 

    def chisquare_fit(self, x, y, yerr = None, xerr = None, func = None, verbose = None, hessian = False, method = 'leastsq', show_correl = True):
        """Perform a fit of this model to the data provided in this function.

        Parameters
        ----------
        x : ArrayLike
            x-values of the data points
        y : ArrayLike
            y-values of the data points
        yerr : ArrayLike
            1-sigma error on the y-values
        xerr : ArrayLike, optional
            1-sigma error on the x-values
        func: function, optional
            Not implemented
        verbose : Bool, optional
            Not implemented
        hessian : bool, optional
            Not implemented
        method : str, optional
            Selects the method used by the :func:`lmfit.minimizer`, by default 'leastsq'.
        show_correl : bool, optional
            `show correlations between fitted parameters in fit message, by default True

        Returns
        -------
        Instance of Fitter (from at satlas2)
        """
        if (func,verbose,hessian) != (None,None,False):
            raise NotImplementedError('Not implemented')
        datasource = Source(x,
                                    y,
                                    yerr=yerr,
                                    name='Fit')
        for model in self.models:
            datasource.addModel(model)
        step_bkg = Step(self.background_params['values'],self.background_params['bounds'], name='bkg')
        self.models.append(step_bkg)
        datasource.addModel(step_bkg)
        f = Fitter()
        f.addSource(datasource)
        f.fit(method = method)
        print(f.reportFit(show_correl = show_correl))
        return f
        
def chisquare_fit(model, x, y, yerr, xerr = None, method = 'leastsq', show_correl = True):
    """Perform a fit of the provided model to the data provided in this function.

        Parameters
        ----------
        x : ArrayLike
            x-values of the data points
        y : ArrayLike
            y-values of the data points
        yerr : ArrayLike
            1-sigma error on the y-values
        xerr : ArrayLike, optional
            1-sigma error on the x-values
        method : str, optional
            Selects the method used by the :func:`lmfit.minimizer`, by default 'leastsq'.
        show_correl : bool, optional
            `show correlations between fitted parameters in fit message, by default True
        
        Returns
        -------
        Instance of Fitter (from at satlas2)
        """
    print('Use satlas2')
    return model.chisquare_fit(x = x, y = y, yerr = yerr, xerr = xerr, method = method, show_correl = show_correl)