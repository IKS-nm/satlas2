"""
Implementation of the base Fitter, Source, Model and Parameter classes

.. moduleauthor:: Wouter Gins <wouter.a.gins@jyu.fi>
"""

from __future__ import annotations

import copy
from typing import Union

import iminuit
import lmfit as lm
import numdifftools as nd
import numpy as np
import scipy.optimize as optimize
from numpy.typing import ArrayLike

from .overwrite import (SATLASHDFBackend, SATLASMinimizer, SATLASSampler,
                        minimize)

__all__ = ['Fitter', 'Source', 'Model', 'Parameter']


def modifiedSqrt(input: ArrayLike) -> ArrayLike:
    output = np.sqrt(input)
    output[input <= 0] = 1
    return output


class Fitter:
    """
    Main class for performing fits and organising data
    """
    def __init__(self):
        super().__init__()
        self.sources = []
        self.pars = {}
        self.bounds = optimize.Bounds([], [])
        self.share = []
        self.shareModel = []
        self.priors = []
        self.expressions = {}
        self.mode = 'source'

    def setExpr(self, parameter_name: Union[list, str],
                parameter_expression: Union[list, str]) -> None:
        """
        Set the expression to be used for the given parameters.
        The given parameter names should be the full description
        i.e. containing the source and model name.

        Note
        ----
        The priority order on expressions is
            1. Expressions given by :func:`~Fitter.setExpr`
            2. Sharing of parameters through :func:`~Fitter.shareParams`
            3. Sharing of parameters through :func:`~Fitter.shareModelParams`

        Parameters
        ----------
        parameter_name: list or str
            Either a single parameter name or a list of them.
        parameter_expression: list or str
            The parameter expression to be associated with parameter_name.
        """
        if isinstance(parameter_name, str):
            parameter_name = [parameter_name]
        if isinstance(parameter_expression, str):
            parameter_expression = [parameter_expression]
        for parameter, expression in zip(parameter_name, parameter_expression):
            self.expressions[parameter] = expression

    def shareParams(self, parameter_name: Union[list, str]) -> None:
        """Add parameters to the list of shared parameters.

        Note
        ----
        The full parameter name should be given.

        Note
        ----
        The priority order on expressions is
            1. Expressions given by :func:`~Fitter.setExpr`
            2. Sharing of parameters through :func:`~Fitter.shareParams`
            3. Sharing of parameters through :func:`~Fitter.shareModelParams`

        Parameters
        ----------
        parameter_name : list or str
            List of parameters or single parameter name.
        """
        try:
            self.share.extend(parameter_name)
        except:
            self.share.append(parameter_name)

    def shareModelParams(self, parameter_name: Union[list, str]) -> None:
        """Add parameters to the list of shared parameters across all
        models with the same name.

        Note
        ----
        The priority order on expressions is
            1. Expressions given by :func:`~Fitter.setExpr`
            2. Sharing of parameters through :func:`~Fitter.shareParams`
            3. Sharing of parameters through :func:`~Fitter.shareModelParams`

        Parameters
        ----------
        parameter_name : list or str
            List of parameters or single parameter name.
        """
        try:
            self.shareModel.extend(parameter_name)
        except:
            self.shareModel.append(parameter_name)

    def setParamPrior(self, source: str, model: str, parameter_name: str,
                      value: float, uncertainty: float) -> None:
        """Set a Gaussian prior on a parameter, mainly intended to
        represent literature values.

        Parameters
        ----------
        source : str
            Name of the datasource in which the parameter is present.
        model : str
            Name of the model in which the parameter is present.
        parameter_name : str
            Name of the parameter.
        value : float
            Central value of the Gaussian
        uncertainty : float
            Standard deviation associated with the value.
        """
        self.priors.append((source, model, parameter_name, value, uncertainty))

    def addSource(self, source: 'Source') -> None:
        """Add a datasource to the Fitter structure

        Parameters
        ----------
        source : Source
            Source to be added to the fitter
        """
        name = source.name
        self.sources.append((name, source))

    def _createParameters(self) -> None:
        """Initialize the parameters from the sources."""
        for name, source in self.sources:
            self.pars[name] = source.params()

    def _createLmParameters(self) -> None:
        """Creates the lmfit parameters."""
        lmpars = lm.Parameters()
        sharing = {}
        sharingModel = {}
        tuples = ()
        for source_name in self.pars.keys(
        ):  # Loop over every datasource in the created parameters
            p = self.pars[source_name]
            for model_name in p.keys(
            ):  # Loop over every model in the datasource
                pars = p[model_name]
                for parameter_name in pars.keys(
                ):  # Loop over every parameter in the model
                    parameter = pars[parameter_name]
                    n = '___'.join([source_name, model_name,
                                    parameter_name])  # Set a unique name
                    parameter.name = '___'.join([source_name, model_name
                                                 ])  # Set a unique identifier
                    if n in self.expressions.keys():
                        expr = self.expressions[n]
                    elif parameter_name in self.share:  # Set the sharing of a variable with EVERY model
                        if parameter_name in sharing.keys(
                        ):  # If not the first instance of a shared variable, get the parameter name
                            expr = sharing[parameter_name]
                        else:
                            sharing[
                                parameter_name] = n  # If the first instance of a shared variable, set it in the sharing dictionary
                            expr = parameter.expr
                    elif parameter_name in self.shareModel:  # Set the sharing of a variable across all models with the SAME NAME
                        if parameter_name in sharingModel.keys(
                        ) and model_name in sharingModel[parameter_name].keys(
                        ):
                            expr = sharingModel[parameter_name][model_name]
                        else:
                            try:
                                sharingModel[parameter_name][model_name] = n
                            except:
                                sharingModel[parameter_name] = {model_name: n}
                            expr = parameter.expr
                    else:
                        expr = parameter.expr
                    tuples += ((n, parameter.value, parameter.vary,
                                parameter.min, parameter.max, expr, None), )
        lmpars.add_many(*tuples)
        self.lmpars = lmpars

    def f(self) -> ArrayLike:
        """Calculate the response of the models in the different sources, stacked horizontally.

        Returns
        -------
        ArrayLike
            Horizontally concatenated response from each source.
        """
        return np.hstack([source.f() for _, source in self.sources])

    def y(self) -> ArrayLike:
        """Stack the data in the different sources, horizontally.

        Returns
        -------
        ArrayLike
            Horizontally concatenated data from each source.
        """
        return np.hstack([source.y for _, source in self.sources])

    def yerr(self) -> ArrayLike:
        """Stack the uncertainty in the different sources, horizontally.

        Returns
        -------
        ArrayLike
            Horizontally concatenated uncertainty from each source.
        """
        return np.hstack([source.yerr() for _, source in self.sources])

    def setParameters(self, params: lm.Parameters) -> None:
        """Set the parameters of the underlying Models
        based on a large Parameters object

        Parameters
        ----------
        params : lm.Parameters
        """
        for p in params.keys():
            if params[p].vary or params[p].expr != None:
                source_name, model_name, parameter_name = p.split('___')
                self.pars[source_name][model_name][
                    parameter_name].value = params[p].value

    def setUncertainties(self, params: lm.Parameters) -> None:
        """Set the uncertainties of the underlying Models
        based on a large Parameters object

        Parameters
        ----------
        params : lm.Parameters
        """
        for p in params.keys():
            source_name, model_name, parameter_name = p.split('___')
            self.pars[source_name][model_name][parameter_name].unc = params[
                p].stderr

    def setCorrelations(self, params: lm.Parameters) -> None:
        """Set the correlations of the underlying Models
        based on a large Parameters object

        Parameters
        ----------
        params : lmfit.Parameters
        """
        for p in params.keys():
            source_name, model_name, parameter_name = p.split('___')
            dictionary = copy.deepcopy(params[p].correl)
            del_keys = []
            try:
                keys = list(dictionary.keys())
                for key in keys:
                    if key.startswith(self.pars[source_name][model_name]
                                      [parameter_name].name):
                        dictionary[key.split('___')[-1]] = dictionary[key]
                    del_keys.append(key)
                for key in del_keys:
                    del dictionary[key]
                self.pars[source_name][model_name][
                    parameter_name].correl = dictionary
            except AttributeError:
                pass

    def resid(self) -> ArrayLike:
        """Calculates the residuals for use in a Gaussian fitting.
        Based on the value of :attr:`Fitter.mode`, a different method is
        used. If :attr:`Fitter.mode` is 'source', the result of :func:`~Fitter.yerr` is used.
        If :attr:`Fitter.mode` is 'combined', the denominator is calculated as

        .. math::
            \sqrt{\\frac{3}{\\frac{1}{y}+\\frac{2}{f(x)}}}
            
        Returns
        -------
        ArrayLike
        """
        model_calcs = self.f()
        if self.mode == 'source':
            resid = (model_calcs - self.temp_y) / self.yerr()
        elif self.mode == 'combined':
            resid = (model_calcs - self.temp_y) / modifiedSqrt(
                3 / (1 / self.temp_y + 2 / model_calcs))
        if np.any(np.isnan(resid)):
            resid[np.isnan(resid)] = np.inf
        return resid

    def gaussianPriorResid(self) -> ArrayLike:
        """Calculates the residual (x-xtrue)/sigma for use
        in a Gaussian prior. The parameters for which this calculates
        the priors are given by :func:`~Fitter.setPrior`.

        Returns
        -------
        ArrayLike
        """
        return np.array([
            (self.pars[source][model][parameter].value - value) / uncertainty
            for source, model, parameter, value, uncertainty in self.priors
        ])

    def residualCalculation(self) -> ArrayLike:
        """Calculates the full residual, based on :func:`~Fitter.resid`
        and :func:`~Fitter.gaussianPriorResid`

        Returns
        -------
        ArrayLike
        """
        return np.hstack([self.resid(), self.gaussianPriorResid()])

    def gaussLlh(self) -> ArrayLike:
        """Calculate the Gaussian likelihood

        Returns
        -------
        ArrayLike
        """
        resid = self.residualCalculation()
        return -0.5 * resid * resid  # Faster than **2

    def poissonLlh(self) -> ArrayLike:
        """Calculate the Poisson likelihood

        Returns
        -------
        ArrayLike
        """
        model_calcs = self.f()
        returnvalue = self.temp_y * np.log(model_calcs) - model_calcs
        returnvalue[model_calcs <= 0] = -np.inf
        return returnvalue

    def llh(self,
            params: lm.Parameters,
            method: str = 'gaussian',
            emcee: bool = False) -> ArrayLike:
        """Calculate the likelihood, based on the parameters and method.
        In case the minimizer uses the emcee package, the array is summed to a single number.
        In case the minimizer uses any other routine, the array is multiplied by -1 to
        obtain the negative likelihood.

        Parameters
        ----------
        params : lm.Parameters
            Parameters for which the likelihood has to be calculated.
        method : str, optional
            Defines either a Gaussian or Poissonian likelihood, by default 'gaussian'.
        emcee : bool, optional
            Toggles the output to be usable by the emcee package, by default False.

        Returns
        -------
        ArrayLike
            An array of the negative loglikelihood (emcee=False) or
            a single number giving the loglikelihood (emcee=True).
        """
        methods = {'gaussian': self.gaussLlh, 'poisson': self.poissonLlh}
        self.setParameters(params)
        returnvalue = methods[method.lower()]()
        if not emcee:
            returnvalue[~np.isfinite(returnvalue)] = -1e99
            returnvalue *= -1
        else:
            returnvalue = np.sum(returnvalue)
        return returnvalue

    def reduction(self, r: ArrayLike) -> float:
        """Reduces the likelihood to a single number. Used by lmfit.

        Parameters
        ----------
        r : ArrayLike
            Array of residuals

        Returns
        -------
        float
            Sum of array of residuals
        """
        return np.sum(r)

    def chisquare(self, params: lm.Parameters) -> ArrayLike:
        """Chisquare optimization function for lmfit.

        Parameters
        ----------
        params : lm.Parameters
            Parameters for which the chisquare has to be calculated

        Returns
        -------
        ArrayLike
            Array of residuals, to be squared and summed by lmfit
        """
        self.setParameters(params)
        return self.residualCalculation()

    def _prepareFit(self):
        self._createParameters()
        self._createLmParameters()

    def reportFit(self,
                  modelpars: Union[lm.Parameters, None] = None,
                  show_correl: bool = False,
                  min_correl: float = 0.1,
                  sort_pars: Union[bool, callable] = False) -> str:
        """Generate a report of the fitting results.

        The report contains the best-fit values for the parameters and their uncertainties and correlations.

        Parameters
        ----------
        modelpars : lmfit.Parameters, optional
            Known Model Parameters
        show_correl : bool, optional
            Whether to show a list of sorted correlations, by default False
        min_correl : float, optional
            Smallest correlation in absolute value to show, by default 0.1
        sort_pars : bool or callable, optional
            Whether to show parameter names sorted in alphanumerical order.
            If False (default), then the parameters will be listed in the
            order they were added to the Parameters dictionary. If callable,
            then this (one argument) function is used to extract a comparison
            key from each list element.

        Returns
        -------
        str
            Multi-line text of fit report.
        """                  
        return lm.fit_report(self.result, modelpars, show_correl, min_correl,
                             sort_pars)

    def fit(self,
            llh: bool=False,
            llh_method: str='gaussian',
            method: str='leastsq',
            mcmc_kwargs: dict={},
            sampler_kwargs: dict={},
            filename: str=None,
            steps: int=1000,
            nwalkers: int=50,
            scale_covar: bool=True) -> None:
        """Perform a fit of the models (added to the sources) to the data in the sources.
        Models in the same source are summed together, models in different sources can be
        linked through their parameters.

        Parameters
        ----------
        llh : bool, optional
            Selects if a chisquare (False) or likelihood fit is performed, by default False.
        llh_method : str, optional
            Selects which likelihood calculation is used, by default 'gaussian'.
        method : str, optional
            Selects the method used by the :func:`lmfit.minimizer`, by default 'leastsq'.
            Set to 'emcee' for random walk.
        mcmc_kwargs : dict, optional
            Dictionary of keyword arguments to be supplied to the MCMC routine
            (see :func:`emcee.EnsembleSampler.sample`), by default {}
        sampler_kwargs : dict, optional
            Dictionary of keyword arguments to be supplied to the :func:`emcee.EnsembleSampler`
            , by default {}
        filename : str, optional
            Filename in which the random walk should be saved, by default None
        steps : int, optional
            Number of steps the random walk should, by default 1000
        nwalkers : int, optional
            Number of walkers to be used in the random walk, by default 50
        scale_covar : bool, optional
            Scale the calculated uncertainties by the root of the reduced
            chisquare, by default True. Set to False when llh is True, since
            the reduced chisquare calculated in this case is not applicable.
        """            
        self.temp_y = self.y()
        self._prepareFit()

        kws = {}
        kwargs = {}
        if llh or method.lower() == 'emcee':
            llh = True
            func = self.llh
            kws['method'] = llh_method
            if method.lower() in ['leastsq', 'least_squares']:
                method = 'slsqp'
        else:
            func = self.chisquare

        if method == 'emcee':
            llh = True
            func = self.llh
            kws['method'] = llh_method
            kws['emcee'] = True
            mcmc_kwargs['skip_initial_state_check'] = True
            if filename is not None:
                sampler_kwargs['backend'] = SATLASHDFBackend(filename)
            else:
                sampler_kwargs['backend'] = None

            kwargs = {
                'mcmc_kwargs': mcmc_kwargs,
                'sampler_kwargs': sampler_kwargs
            }

            kwargs['sampler'] = SATLASSampler
            kwargs['steps'] = steps
            kwargs['nwalkers'] = nwalkers
            kwargs['nan_policy'] = 'propagate'
        if llh:
            scale_covar = False

        reduce_fcn = self.reduction
        self.result = minimize(func,
                               self.lmpars,
                               method=method,
                               kws=kws,
                               reduce_fcn=reduce_fcn,
                               scale_covar=scale_covar,
                               **kwargs)
        del self.temp_y
        self.updateInfo()

    def revertFit(self):
        params = self.result.init_values
        for p in params.keys():
            source_name, model_name, parameter_name = p.split('___')
            self.pars[source_name][model_name][parameter_name].value = params[
                p]
        self._prepareFit()
        self.setParameters(self.lmpars)

    def readWalk(self, filename):
        reader = SATLASHDFBackend(filename)
        # var_names = list(reader.labels)
        data = reader.get_chain(flat=False)
        try:
            self.result = SATLASMinimizer(self.llh, self.lmpars).process_walk(
                self.lmpars, data)
        except AttributeError:
            self._prepareFit()
            self.result = SATLASMinimizer(self.llh, self.lmpars).process_walk(
                self.lmpars, data)
        self.updateInfo()

    def updateInfo(self):
        self.lmpars = self.result.params
        self.setParameters(self.result.params)
        self.setUncertainties(self.result.params)
        self.setCorrelations(self.result.params)
        self.nvarys = self.result.nvarys
        try:
            self.nfree = self.result.nfree
            self.ndata = self.result.ndata
            self.chisqr = self.result.chisqr
            self.redchi = self.result.redchi
        except:
            pass
        self.updateFitInfoSources()

    def updateFitInfoSources(self):
        for source_name, source in self.sources:
            source.nvarys = self.nvarys
            try:
                source.chisqr = self.chisqr
                source.ndata = self.ndata
                source.nfree = self.nfree
                source.redchi = self.redchi
            except:
                pass


class Source:
    def __init__(self, x, y, yerr, xerr=None, name=None):
        super().__init__()
        self.x = x
        self.y = y
        self.xerr = xerr
        self.yerr_data = yerr
        if name is not None:
            self.name = name
        self.models = []
        self.derivative = nd.Derivative(self.evaluate)

    def addModel(self, model, name=None):
        if name is None:
            name = model.name
        self.models.append((name, model))

    def params(self):
        params = {}
        for name, model in self.models:
            params[name] = model.params
        return params

    def f(self):
        for _, model in self.models:
            try:
                f += model.f(self.x)
            except UnboundLocalError:
                f = model.f(self.x)
        return f

    def evaluate(self, x):
        for name, model in self.models:
            try:
                f += model.f(x)
            except UnboundLocalError:
                f = model.f(x)
        return f

    def yerr(self):
        err = None
        if not callable(self.yerr_data):
            err = self.yerr_data
        else:
            err = self.yerr_data(self.f())
        if self.xerr is not None:
            xerr = self.derivative(self.x) * self.xerr
            err = (err * err + xerr * xerr)**0.5
        return err


class Model:
    def __init__(self, prefunc=None, name=None):
        super().__init__()
        self.name = name
        self.prefunc = prefunc
        self.params = {}
        self.xtransformed = None
        self.xhashed = None

    def transform(self, x):
        if callable(self.prefunc):
            hashed = x.data.tobytes()
            if hashed == self.xhashed:
                x = self.xtransformed
            else:
                x = self.prefunc(x)
                self.xtransformed = x
                self.xhashed = hashed
        return x

    def setTransform(self, func):
        self.prefunc = func

    def setBounds(self, name, bounds):
        if name in self.params.keys():
            self.params[name].min = min(bounds)
            self.params[name].max = max(bounds)

    def setVary(self, name, vary):
        if name in self.params.keys():
            self.params[name].vary = vary

    def setExpr(self, name, expr):
        if name in self.params.keys():
            self.params[name].expr = expr

    def addParameter(self, **kwargs):
        name = kwargs.pop('name')
        self.params[name] = Parameter(**kwargs)

    def f(self, x) -> float:
        raise NotImplemented


class Parameter:
    def __init__(self, value=0, min=-np.inf, max=np.inf, vary=True, expr=None):
        super().__init__()
        self.value = value
        self.min = min
        self.max = max
        self.vary = vary
        self.expr = expr
        self.unc = None
        self.correl = None
        self.name = ''

    def __repr__(self):
        return '{}+/-{} ({} max, {} min, vary={}, correl={})'.format(
            self.value, self.unc, self.max, self.min, self.vary, self.correl)
