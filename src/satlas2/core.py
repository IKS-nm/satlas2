"""
Implementation of the base Fitter, Source, Model and Parameter classes

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple, Union

import lmfit as lm
import numdifftools as nd
import numpy as np
import pandas as pd
import scipy.optimize as optimize
from numpy.typing import ArrayLike

from .overwrite import (
    SATLASHDFBackend,
    SATLASMinimizer,
    SATLASSampler,
    minimize,
)

__all__ = ["Fitter", "Source", "Model", "Parameter"]


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
        self.priors = {}
        self.expressions = {}
        self.mode = "source"
        self.llhmethods = {
            "gaussian": self.gaussLlh,
            "poisson": self.poissonLlh,
            "custom": self.customLlh,
        }

    def setExpr(
        self,
        parameter_name: Union[list, str],
        parameter_expression: Union[list, str],
    ) -> None:
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

    def removeExpr(self, parameter_name: Union[list, str]) -> None:
        """
        Remove the expression for the given parameters.

        Parameters
        ----------
        parameter_name : list or str
            Either a single parameter name or a list of them.
        """
        if isinstance(parameter_name, str):
            parameter_name = [parameter_name]
        for p in parameter_name:
            try:
                del self.expressions[p]
            except:
                pass

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

    def removeShareParams(self, parameter_name: Union[list, str]) -> None:
        """Removed shared parameter.

        Note
        ----
        The full parameter name should be given.

        Parameters
        ----------
        parameter_name : Union[list, str]
            List of parameters or single parameter name.
        """
        if isinstance(parameter_name, str):
            parameter_name = [parameter_name]
        for p in parameter_name:
            try:
                self.share.remove(p)
            except ValueError:
                pass

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

    def removeShareModelParams(self, parameter_name: Union[list, str]) -> None:
        """Remove parameters shared across all models with the same name.

        Parameters
        ----------
        parameter_name : Union[list, str]
            List of parameters or single parameter name.
        """
        if isinstance(parameter_name, str):
            parameter_name = [parameter_name]
        for p in parameter_name:
            try:
                self.shareModel.remove(p)
            except ValueError:
                pass

    def setParamPrior(
        self,
        source: str,
        model: str,
        parameter_name: str,
        value: float,
        uncertainty: float,
    ) -> None:
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
        self.priors["___".join([source, model, parameter_name])] = (
            value,
            uncertainty,
        )

    def removeParamPrior(
        self, source: str, model: str, parameter_name: str
    ) -> None:
        """Removes a prior set on a parameter.

        Parameters
        ----------
        source : str
            Name of the datasource in which the parameter is present.
        model : str
            Name of the model in which the parameter is present.
        parameter_name : str
            Name of the parameter.
        """
        del self.priors["___".join([source, model, parameter_name])]

    def removeAllPriors(self):
        """Removes all priors on parameters."""
        self.priors = {}

    def addSource(self, source: Source) -> None:
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
        for (
            source_name
        ) in (
            self.pars.keys()
        ):  # Loop over every datasource in the created parameters
            p = self.pars[source_name]
            for (
                model_name
            ) in p.keys():  # Loop over every model in the datasource
                pars = p[model_name]
                for (
                    parameter_name
                ) in pars.keys():  # Loop over every parameter in the model
                    parameter = pars[parameter_name]
                    n = "___".join(
                        [source_name, model_name, parameter_name]
                    )  # Set a unique name
                    parameter.name = "___".join(
                        [source_name, model_name]
                    )  # Set a unique identifier
                    if n in self.expressions.keys():
                        expr = self.expressions[n]
                    elif (
                        parameter_name in self.share
                    ):  # Set the sharing of a variable with EVERY model
                        if (
                            parameter_name in sharing.keys()
                        ):  # If not the first instance of a shared variable, get the parameter name
                            expr = sharing[parameter_name]
                        else:
                            sharing[
                                parameter_name
                            ] = n  # If the first instance of a shared variable, set it in the sharing dictionary
                            expr = parameter.expr
                    elif (
                        parameter_name in self.shareModel
                    ):  # Set the sharing of a variable across all models with the SAME NAME
                        if (
                            parameter_name in sharingModel.keys()
                            and model_name
                            in sharingModel[parameter_name].keys()
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
                    tuples += (
                        (
                            n,
                            parameter.value,
                            parameter.vary,
                            parameter.min,
                            parameter.max,
                            expr,
                            None,
                        ),
                    )
        lmpars.add_many(*tuples)
        self.lmpars = lmpars

    def f(self) -> ArrayLike:
        """Calculate the response of the models in the different sources,
        stacked horizontally.

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

    def getSourceAttr(self, attr: str) -> ArrayLike:
        """Stack the giveen attributed in the different sources, horizontally.

        Parameters
        ----------
        attr : str
            Attribute of the sources to be retrieved.

        Returns
        -------
        ArrayLike
            Horizontally concatenated attribute from each source.
        """
        return np.hstack([getattr(source, attr) for _, source in self.sources])

    def setParameters(self, params: lm.Parameters) -> None:
        """:meta private:
        Set the parameters of the underlying Models
        based on a large Parameters object

        Parameters
        ----------
        params : lm.Parameters
        """
        for p in params.keys():
            if params[p].vary or params[p].expr != None:
                source_name, model_name, parameter_name = p.split("___")
                self.pars[source_name][model_name][
                    parameter_name
                ].value = params[p].value

    def setUncertainties(self, params: lm.Parameters) -> None:
        """:meta private:
        Set the uncertainties of the underlying Models
        based on a large Parameters object

        Parameters
        ----------
        params : lm.Parameters
        """
        for p in params.keys():
            source_name, model_name, parameter_name = p.split("___")
            self.pars[source_name][model_name][parameter_name].unc = params[
                p
            ].stderr

    def setCorrelations(self, params: lm.Parameters) -> None:
        """:meta private:
        Set the correlations of the underlying Models
        based on a large Parameters object

        Parameters
        ----------
        params : lmfit.Parameters
        """
        for p in params.keys():
            source_name, model_name, parameter_name = p.split("___")
            dictionary = copy.deepcopy(params[p].correl)
            del_keys = []
            try:
                keys = list(dictionary.keys())
                for key in keys:
                    if key.startswith(
                        self.pars[source_name][model_name][parameter_name].name
                    ):
                        dictionary[key.split("___")[-1]] = dictionary[key]
                    del_keys.append(key)
                for key in del_keys:
                    del dictionary[key]
                self.pars[source_name][model_name][
                    parameter_name
                ].correl = dictionary
            except AttributeError:
                pass

    def resid(self) -> ArrayLike:
        """:meta private:
        Calculates the residuals for use in a Gaussian fitting.
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
        if self.mode == "source":
            resid = (model_calcs - self.temp_y) / self.yerr()
        elif self.mode == "combined":
            resid = (model_calcs - self.temp_y) / modifiedSqrt(
                3 / (1 / self.temp_y + 2 / model_calcs)
            )
        if np.any(np.isnan(resid)):
            resid[np.isnan(resid)] = np.inf
        return resid

    def gaussianPriorResid(self) -> ArrayLike:
        """:meta private:
        Calculates the residual (x-xtrue)/sigma for use
        in a Gaussian prior. The parameters for which this calculates
        the priors are given by :func:`~Fitter.setParamPrior`.

        Returns
        -------
        ArrayLike
        """
        returnval = []
        for key in self.priors.keys():
            source, model, parameter = key.split("___")
            lit, unc = self.priors[key]
            returnval.append(
                (self.pars[source][model][parameter].value - lit) / unc
            )
        return np.array(returnval)

    def residualCalculation(self) -> ArrayLike:
        """:meta private:
        Calculates the full residual, based on :func:`~Fitter.resid`
        and :func:`~Fitter.gaussianPriorResid`

        Returns
        -------
        ArrayLike
        """
        return np.hstack([self.resid(), self.gaussianPriorResid()])

    def gaussLlh(self) -> ArrayLike:
        """:meta private:
        Calculate the Gaussian likelihood

        Returns
        -------
        ArrayLike
        """
        resid = self.residualCalculation()
        return -0.5 * resid * resid  # Faster than **2

    def poissonLlh(self) -> ArrayLike:
        """:meta private:
        Calculate the Poisson likelihood

        Returns
        -------
        ArrayLike
        """
        model_calcs = self.f()
        returnvalue = self.temp_y * np.log(model_calcs) - model_calcs
        returnvalue[model_calcs <= 0] = -np.inf
        priors = self.gaussianPriorResid()
        if len(priors) > 1:
            priors = -0.5 * priors * priors
            returnvalue = np.append(returnvalue, priors)
        return returnvalue

    def customLlh(self):
        """Calculate a custom likelihood."""
        raise NotImplementedError

    def llh(
        self,
        params: lm.Parameters,
        method: str = "gaussian",
        emcee: bool = False,
    ) -> ArrayLike:
        """:meta private:
        Calculate the likelihood, based on the parameters and method.
        In case the minimizer uses the emcee package, the array is summed to a single number.
        In case the minimizer uses any other routine, the array is multiplied by -1 to
        obtain the negative likelihood.

        Parameters
        ----------
        params : lm.Parameters
            Parameters for which the likelihood has to be calculated.
        method : str, optional
            Defines either a Gaussian, Poissonian or custom likelihood, by default 'gaussian'.
        emcee : bool, optional
            Toggles the output to be usable by the emcee package, by default False.

        Returns
        -------
        ArrayLike
            An array of the negative loglikelihood (emcee=False) or
            a single number giving the loglikelihood (emcee=True).
        """
        self.setParameters(params)
        returnvalue = self.llhmethods[method.lower()]()
        if not emcee:
            returnvalue[~np.isfinite(returnvalue)] = -1e99
            returnvalue *= -1
        else:
            returnvalue = np.sum(returnvalue)
        return returnvalue

    def reductionSum(self, r: ArrayLike) -> float:
        """:meta private:
        Reduces the likelihood to a single number. Used by lmfit.

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

    def reductionSSum(self, r: ArrayLike) -> float:
        """:meta private:
        Reduces the likelihood to a single number. Used by lmfit.

        Parameters
        ----------
        r : ArrayLike
            Array of residuals

        Returns
        -------
        float
            Sum of squares of array of residuals
        """
        return np.sum(r * r)

    def chisquare(self, params: lm.Parameters) -> ArrayLike:
        """:meta private:
        Chisquare optimization function for lmfit.

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
        """:meta private:"""
        self._createParameters()
        self._createLmParameters()

    def revertFit(self):
        """Reverts the parameter values to the original values."""
        params = self.result.init_values
        for p in params.keys():
            source_name, model_name, parameter_name = p.split("___")
            self.pars[source_name][model_name][parameter_name].value = params[
                p
            ]
        self._prepareFit()
        self.setParameters(self.lmpars)

    def fit(
        self,
        llh: bool = False,
        llh_method: str = "gaussian",
        method: str = "leastsq",
        mcmc_kwargs: dict = {},
        sampler_kwargs: dict = {},
        filename: Optional[str] = None,
        overwrite: bool = True,
        nwalkers: int = 50,
        steps: int = 1000,
        convergence: bool = False,
        convergence_iter: int = 50,
        convergence_tau: float = 0.05,
        scale_covar: bool = True,
        iter_cb: Optional[callable] = None,
    ) -> None:
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
        overwrite: bool, optional
            If True, the generated file is overwritten. If False, the number of walkers and the
            last position is taken from the saved file. By default True.
        nwalkers : int, optional
            Number of walkers to be used in the random walk, by default 50
        steps : int, optional
            Number of steps the random walk should take, by default 1000
        convergence : bool, optional
            Controls automatically stopping of the random walk based on the
            autocorrelation criteria, by default False.
        convergence_iter : int, optional
            Factor by which the number of steps taken should be greater than
            the autocorrelation time, by default 50.
        convergence_tau : float, optional
            Relative value within which subsequent autocorrelation estimates
            should lie for convergence, by default 0.05.
        scale_covar : bool, optional
            Scale the calculated uncertainties by the root of the reduced
            chisquare, by default True. Set to False when llh is True, since
            the reduced chisquare calculated in this case is not applicable.
        """
        self.temp_y = self.y()
        self._prepareFit()

        kws = {}
        kwargs = {}
        kwargs["iter_cb"] = iter_cb
        reduce_fcn = self.reductionSum
        if llh or method.lower() == "emcee":
            llh = True
            func = self.llh
            kws["method"] = llh_method
            if method.lower() in ["leastsq", "least_squares"]:
                method = "slsqp"
        else:
            func = self.chisquare
            reduce_fcn = self.reductionSSum

        if method == "emcee":
            llh = True
            func = self.llh
            kws["method"] = llh_method
            kws["emcee"] = True
            mcmc_kwargs["skip_initial_state_check"] = True
            import os.path

            kwargs["load"] = os.path.isfile(filename) and (not overwrite)
            if filename is not None:
                sampler_kwargs["backend"] = SATLASHDFBackend(filename)
            else:
                sampler_kwargs["backend"] = None

            kwargs["mcmc_kwargs"] = mcmc_kwargs
            kwargs["sampler_kwargs"] = sampler_kwargs

            kwargs["sampler"] = SATLASSampler
            kwargs["steps"] = steps
            kwargs["nwalkers"] = nwalkers
            kwargs["nan_policy"] = "propagate"
            kwargs["convergence"] = convergence
            kwargs["convergence_tau"] = convergence_tau
            kwargs["convergence_iter"] = convergence_iter
        if llh:
            scale_covar = False

        self.result = minimize(
            func,
            self.lmpars,
            method=method,
            kws=kws,
            reduce_fcn=reduce_fcn,
            scale_covar=scale_covar,
            **kwargs,
        )
        del self.temp_y
        self.updateInfo()

    def reportFit(
        self,
        modelpars: Optional[lm.Parameters] = None,
        show_correl: bool = False,
        min_correl: float = 0.1,
        sort_pars: Union[bool, callable] = False,
    ) -> str:
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
        return lm.fit_report(
            self.result, modelpars, show_correl, min_correl, sort_pars
        )

    def createResultDataframe(self) -> pd.DataFrame:
        """Generates a dataframe containing all information about the parameters
        after a fit.

        Returns
        -------
        pd.DataFrame"""
        data = [
            [
                p.split("___")[0],
                p.split("___")[1],
                p.split("___")[2],
                self.result.params[p].value,
                self.result.params[p].stderr,
                self.result.params[p].min,
                self.result.params[p].max,
                self.result.params[p].expr,
                self.result.params[p].vary,
            ]
            for p in self.result.params
        ]
        columns = [
            "Source",
            "Model",
            "Parameter",
            "Value",
            "Stderr",
            "Minimum",
            "Maximum",
            "Expression",
            "Vary",
        ]
        df = pd.DataFrame(data=data, columns=columns)
        return df

    def createMetadataDataframe(self) -> pd.DataFrame:
        """Generates a dataframe containing the fitting information and
        statistics.

        Returns
        -------
        pd.DataFrame"""
        columns = [
            "Source",
            "Fitting method",
            "Message",
            "Function evaluations",
            "Data points",
            "Variables",
            "Chisquare",
            "Redchi",
            "Aic",
            "Bic",
        ]
        source = [name for (name, s) in self.sources]
        source = ", ".join(source)
        data = [
            [
                source,
                self.result.method,
                self.result.message,
                self.result.nfev,
                self.result.ndata,
                self.result.nvarys,
                self.result.chisqr,
                self.result.redchi,
                self.result.aic,
                self.result.bic,
            ]
        ]
        df = pd.DataFrame(data=data, columns=columns)
        return df

    def readWalk(self, filename: str, burnin: Optional[int] = 0):
        """Read and process the h5 file containing the results of a random walk.
        The parameter values and uncertainties are extracted from the walk.

        Parameters
        ----------
        filename : str
            Filename of the random walk results.
        burnin: Optional[int]
            Optional amount of steps to remove from the start, defaults to 0.
        """
        reader = SATLASHDFBackend(filename)
        # var_names = list(reader.labels)
        data = reader.get_chain(flat=False, discard=burnin)
        try:
            self.result = SATLASMinimizer(self.llh, self.lmpars).process_walk(
                self.lmpars, data
            )
        except AttributeError:
            self._prepareFit()
            self.result = SATLASMinimizer(self.llh, self.lmpars).process_walk(
                self.lmpars, data
            )
        self.updateInfo()

    def updateInfo(self):
        """:meta private:"""
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
        """:meta private:"""
        for _, source in self.sources:
            source.nvarys = self.nvarys
            try:
                source.chisqr = self.chisqr
                source.ndata = self.ndata
                source.nfree = self.nfree
                source.redchi = self.redchi
            except:
                pass

    def evaluateOverWalk(
        self,
        filename: str,
        burnin: int = 0,
        x: Optional[ArrayLike] = None,
        evals: int = 0,
    ) -> Tuple[list, list]:
        """The parameters saved in the h5 file are evaluated
        in the models a specific number of times. From these evaluations, the
        16, 50 and 84 percentiles (corresponding to the 1-sigma band) are calculated.

        Parameters
        ----------
        filename : str
            Filename of the random walk results.
        burnin : int, optional
            Amount of steps to skip, by default 0
        x : ArrayLike, optional
            Evaluation points for the model, defaults to the datapoints in Source
        evals : int, optional
            Number of selected parameter values, defaults to using all values

        Returns
        -------
        Tuple of lists
            A tuple with, as the first element, a list of arrays x-values for
            which the band has been evaluated. Each source contributes an array.
            The second element is a list of 2D arrays, one for each source. The
            first row is the sigma- boundary, the second row is the median value
            and the third row is the sigm+ boundary.
        """
        reader = SATLASHDFBackend(filename)
        var_names = list(reader.labels)
        data = reader.get_chain(flat=False, discard=burnin)
        flatchain = data.reshape((-1, len(var_names)))
        if x is None:
            method = "f"
            args = ()
        else:
            method = "evaluate"
            args = (x,)
        if evals > 0:
            if evals < flatchain.shape[0]:
                choices = np.random.choice(flatchain.shape[0], evals)
                flatchain = flatchain[choices]
        else:
            pass
        try:
            names = [p for p in self.lmpars.keys()]
        except:
            self._prepareFit()
            names = [p for p in self.lmpars.keys()]
        common = [
            (i, name.split("___"))
            for i, name in enumerate(var_names)
            if name in names
        ]
        bands = []
        X = []
        for sample in flatchain:
            for column, splitname in common:
                source, model, parameter = splitname
                self.pars[source][model][parameter].value = sample[column]
            for i, (_, source) in enumerate(self.sources):
                try:
                    bands[i] = np.vstack(
                        [bands[i], getattr(source, method)(*args)]
                    )
                except Exception as e:
                    bands.append(getattr(source, method)(*args))
                if len(args) > 0:
                    X.append(args[0])
                else:
                    X.append(source.x)
        for i, band in enumerate(bands):
            q = np.percentile(band, [16, 84], axis=0)
            bands[i] = q
        median = np.percentile(flatchain, 50, axis=0)
        for column, splitname in common:
            source, model, parameter = splitname
            self.pars[source][model][parameter].value = median[column]
        for i, (_, source) in enumerate(self.sources):
            bands[i] = np.vstack(
                [bands[i][0], getattr(source, method)(*args), bands[i][1]]
            )
        return X, bands


class Source:
    """Initializes a source of data

    Parameters
    ----------
    x : ArrayLike
        x values of the data
    y : ArrayLike
        y values of the data
    yerr : Union[ArrayLike, callable]
        The yerr of the data, either an array for fixed uncertainties
        or a callable to be applied to the result of the models in the source.
    name : str
        The name given to the source. This must be a unique value!
    xerr : ArrayLike, optional
        If enlargement of the yerr with the xerr is required, supply this, by default None.
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        yerr: Union[ArrayLike, callable],
        name: str,
        xerr: ArrayLike = None,
        **kwargs,
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.xerr = xerr
        self.yerr_data = yerr
        if name is not None:
            self.name = name
        self.models = []
        self.derivative = nd.Derivative(self.evaluate)
        for k in kwargs.keys():
            setattr(self, k, kwargs[k])

    def addModel(self, model: "Model"):
        """Add a model to the Source

        Parameters
        ----------
        model : Model
            The Model to be added to the source. Multiple models give, as a result,
            the sum of the individual models
        """
        self.models.append((model.name, model))

    def params(self) -> dict:
        """:meta private:"""
        params = {}
        for name, model in self.models:
            params[name] = model.params
        return params

    def f(self) -> ArrayLike:
        """Returns the sum of the evaluation of all models in the x-coordinates defined in the source.

        Returns
        -------
        ArrayLike
        """
        for _, model in self.models:
            try:
                f += model.f(self.x)
            except UnboundLocalError:
                f = model.f(self.x)
        return f

    def evaluate(self, x: ArrayLike) -> ArrayLike:
        """Evaluates all models in the given points and returns the sum.

        Parameters
        ----------
        x : ArrayLike
            Points in which the models have to be evaluated

        Returns
        -------
        ArrayLike
        """
        for _, model in self.models:
            try:
                f += model.f(x)
            except UnboundLocalError:
                f = model.f(x)
        return f

    def yerr(self):
        """:meta private:"""
        err = None
        if not callable(self.yerr_data):
            err = self.yerr_data
        else:
            err = self.yerr_data(self.f())
        if self.xerr is not None:
            xerr = self.derivative(self.x) * self.xerr
            err = (err * err + xerr * xerr) ** 0.5
        return err


class Model:
    """Base Model class

    Parameters
    ----------
    name : str
        Name given to the model
    prefunc : callable, optional
        Transformation function to be applied to the
        evaluation points before evaluating the model, by default None
    """

    def __init__(self, name: str, prefunc: Optional[callable] = None):
        super().__init__()
        self.name = name
        self.prefunc = prefunc
        self.params = {}
        self.xtransformed = None
        self.xhashed = None

    def transform(self, x: ArrayLike) -> ArrayLike:
        """:meta private:

        Parameters
        ----------
        x : ArrayLike
            Evaluation points

        Returns
        -------
        ArrayLike
        """
        if callable(self.prefunc):
            hashed = x.data.tobytes()
            if hashed == self.xhashed:
                x = self.xtransformed
            else:
                x = self.prefunc(x)
                self.xtransformed = x
                self.xhashed = hashed
        return x

    def setTransform(self, func: callable):
        """Set the transformation for the pre-evaluation.

        Parameters
        ----------
        func : callable
        """
        self.prefunc = func

    def f(self, x: ArrayLike) -> float:
        """Evaluates the model in the given points.

        Parameters
        ----------
        x : ArrayLike
            Points in which the model has to be evaluated

        Returns
        -------
        ArrayLike
        """
        raise NotImplemented


class Parameter:
    """:meta private:"""

    def __init__(self, value=0, min=-np.inf, max=np.inf, vary=True, expr=None):
        super().__init__()
        self.value = value
        self.min = min
        self.max = max
        self.vary = vary
        self.expr = expr
        self.unc = 0
        self.correl = {}
        self.name = ""

    def representation(self):
        if self.vary:
            if self.unc is None:
                return "{:.2g}".format(self.value)
            else:
                return "{:.2g}+/-{:.2g}".format(self.value, self.unc)
        else:
            return "{:.2g} (fixed)".format(self.value)

    def __repr__(self):
        return "{}+/-{} ({} max, {} min, vary={}, correl={})".format(
            self.value, self.unc, self.max, self.min, self.vary, self.correl
        )
