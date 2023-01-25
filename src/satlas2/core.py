"""
Implementation of the base Fitter, Source, Model and Parameter classes

.. moduleauthor:: Wouter Gins <wouter.a.gins@jyu.fi>
"""

import numpy as np
import scipy.optimize as optimize
import lmfit as lm
import emcee
import numdifftools as nd
import copy
import iminuit
from .overwrite import SATLASSampler, SATLASHDFBackend, minimize, SATLASMinimizer

__all__ = ['Fitter', 'Source', 'Model', 'Parameter']


def modifiedSqrt(input):
    output = np.sqrt(input)
    output[input <= 0] = 0.5
    # output[input <= 0] = 1
    return output


class Fitter:
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

    def setExpr(self, parameter_name, parameter_expression):
        if isinstance(parameter_name, str):
            parameter_name = [parameter_name]
        if isinstance(parameter_expression, str):
            parameter_expression = [parameter_expression]
        for parameter, expression in zip(parameter_name, parameter_expression):
            self.expressions[parameter] = expression

    def shareParams(self, parameter_name):
        """Add parameters to be shared across all models."""
        try:
            self.share.extend(parameter_name)
        except:
            self.share.append(parameter_name)

    def shareModelParams(self, parameter_name):
        """Add parameters to be shared across all models with the same name."""
        try:
            self.shareModel.extend(parameter_name)
        except:
            self.shareModel.append(parameter_name)

    def setParamPrior(self, source, model, parameter_name, value, uncertainty):
        self.priors.append((source, model, parameter_name, value, uncertainty))

    def addSource(self, source, name=None):
        """Add a datasource to the Fitter."""
        if name is None:
            name = source.name
        self.sources.append((name, source))

    def createParameters(self):
        """Initialize the parameters."""
        for name, source in self.sources:
            self.pars[name] = source.params()

    def createMinuit(self):
        self.createParameters()
        self.setParameterMapping()
        m = iminuit.Minuit(self,
                           self.createParameterList(),
                           name=self.parameter_names)
        return m

    def createBounds(self):
        lower = []
        upper = []
        for source_name in self.pars.keys():
            p = self.pars[source_name]
            for model_name in p.keys():
                pars = p[model_name]
                for parameter_name in pars.keys():
                    parameter = pars[parameter_name]
                    if not parameter.vary:
                        l = parameter.value
                        u = parameter.value
                    else:
                        l = parameter.min
                        u = parameter.max
                    lower.append(l)
                    upper.append(u)
        self.bounds = optimize.Bounds(lower, upper)

    def createLmParameters(self):
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

    def createParameterList(self):
        x = []
        for source_name in self.pars.keys():
            p = self.pars[source_name]
            for model_name in p.keys():
                pars = p[model_name]
                for parameter_name in pars.keys():
                    if pars[parameter_name].vary:
                        x.append(pars[parameter_name].value)
        return x

    def setParameterMapping(self):
        self.mapping = {}
        self.parameter_names = []
        i = 0
        for source_name in self.pars.keys():
            p = self.pars[source_name]
            for model_name in p.keys():
                pars = p[model_name]
                for parameter_name in pars.keys():
                    if pars[parameter_name].vary:
                        self.mapping[
                            i] = source_name, model_name, parameter_name
                        self.parameter_names.append('_'.join(
                            [source_name, model_name, parameter_name]))
                        i += 1

    def __call__(self, params):
        for i, value in enumerate(params):
            source_name, model_name, parameter_name = self.mapping[i]
            self.pars[source_name][model_name][parameter_name].value = value
        resid = self.residualCalculation()
        return np.sum(resid * resid)

    def f(self):
        f = []
        for name, source in self.sources:
            f.append(source.f())
        return np.hstack(f)

    def y(self):
        y = []
        for _, source in self.sources:
            y.append(source.y)
        return np.hstack(y)

    def yerr(self):
        yerr = []
        for _, source in self.sources:
            yerr.append(source.yerr())
        return np.hstack(yerr)

    def setParameters(self, params):
        for p in params.keys():
            source_name, model_name, parameter_name = p.split('___')
            self.pars[source_name][model_name][parameter_name].value = params[
                p].value

    def setUncertainties(self, params):
        for p in params.keys():
            source_name, model_name, parameter_name = p.split('___')
            self.pars[source_name][model_name][parameter_name].unc = params[
                p].stderr

    def setCorrelations(self, params):
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

    def resid(self):
        model_calcs = self.f()
        if self.mode == 'source':
            resid = (model_calcs - self.temp_y) / self.yerr()
        elif self.mode == 'combined':
            resid = (model_calcs - self.temp_y) / modifiedSqrt(
                3 / (1 / self.temp_y + 2 / model_calcs))
        if np.any(np.isnan(resid)):
            return np.inf
        else:
            return resid

    def gaussianPriors(self):
        return [
            (self.pars[source][model][parameter].value - value) / uncertainty
            for source, model, parameter, value, uncertainty in self.priors
        ]

    def gaussLlh(self):
        resid = self.residualCalculation()
        # print('here')
        # raise ImportError
        # print('here')
        return 0.5 * resid * resid  # Faster than **2

    def poissonLlh(self):
        model_calcs = self.f()
        returnvalue = self.temp_y * np.log(model_calcs) - model_calcs
        returnvalue[model_calcs<=0] = -1e20
        return -returnvalue

    def poissonChisq(self):
        model_calcs = self.f(y)
        return 2 * (model_calcs - self.temp_y +
                    self.temp_y * np.log(self.temp_y / model_calcs))

    def llh(self, params, method='gaussian', emcee=False):
        methods = {'gaussian': self.gaussLlh, 'poisson': self.poissonLlh}
        self.setParameters(params)
        returnvalue = methods[method.lower()]()
        # returnvalue = np.sum(methods[method.lower()]())
        # if not np.isfinite(returnvalue):
        #     returnvalue = -1e99
        # if not emcee:
        #     returnvalue *= -1
        return returnvalue

    def reduction(self, r):
        return np.sum(r)

    def callback(self, params, iter, resid, *args, **kwargs):
        return None

    def residualCalculation(self):
        resid = self.resid()
        priors = self.gaussianPriors()
        if len(priors) > 0:
            resid = np.append(resid, priors)
        return resid

    def residualCalculationWithParams(self, params):
        resid = self.residParams(params)
        priors = self.gaussianPriors()
        if len(priors) > 0:
            resid = np.append(resid, priors)
        return resid

    def optimizeFunc(self, params):
        self.setParameters(params)
        return self.residualCalculation()

    def prepareFit(self):
        self.createParameters()
        self.createBounds()
        self.createLmParameters()

    def reportFit(self):
        return lm.fit_report(self.result)

    def fit(self,
            prepFit=True,
            llh_selected=False,
            llh_method='gaussian',
            method='leastsq',
            kws={},
            mcmc_kwargs={},
            sampler_kwargs={},
            filename=None,
            steps=1000,
            nwalkers=50,
            scale_covar=True):
        self.temp_y = self.y()
        if prepFit:
            self.prepareFit()
        if llh_method.lower() == 'poisson':
            llh_selected = True

        kws = {}
        kwargs = {}
        if llh_selected or method.lower() == 'emcee':
            llh_selected = True
            func = self.llh
            kws['method'] = llh_method
            if method.lower() in ['leastsq', 'least_squares']:
                method = 'nelder'
        else:
            func = self.optimizeFunc

        if method == 'emcee':
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

        reduce_fcn = self.reduction
        self.result = minimize(func,
                               self.lmpars,
                               method=method,
                               iter_cb=self.callback,
                               kws=kws,
                               reduce_fcn=reduce_fcn,
                               scale_covar=scale_covar,
                               **kwargs)
        if llh_selected:
            self.llh_result = self.llh(self.result.params, method=llh_method)
            self.nllh_result = self.llh(self.result.params, method=llh_method)
        else:
            self.llh_result = None
            self.nllh_result = None
        del self.temp_y
        self.updateInfo()

    def readWalk(self, filename):
        reader = SATLASHDFBackend(filename)
        var_names = list(reader.labels)
        data = reader.get_chain(flat=False)
        try:
            self.result = SATLASMinimizer(self.llh, self.lmpars).process_walk(
                self.lmpars, data)
        except AttributeError:
            self.prepareFit()
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
        # if self.yerr_data == 1:
        #     self.yerr_data = np.ones(self.x.shape)
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
            xerr = self.derivative(self.x)*self.xerr
            err = (err*err + xerr*xerr)**0.5
        return err


class Model:
    def __init__(self, prefunc=None, name=None, pretransform=True):
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

    def params(self):
        return {}

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
