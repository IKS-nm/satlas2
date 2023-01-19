"""
Implementation of the base Fitter, Source, Model and Parameter classes

.. moduleauthor:: Wouter Gins <wouter.a.gins@jyu.fi>
"""

import numpy as np
import scipy.optimize as optimize
import lmfit as lm
import emcee
import copy
import iminuit
from .overwrite import SATLASSampler, SATLASHDFBackend, minimize, SATLASMinimizer

__all__ = ['Fitter', 'Source', 'Model', 'Parameter']

class Cost:
    def __init__(self):
        super().__init__()
        self.sources = []
        self.pars = {}
        self.bounds = optimize.Bounds([], [])
        self.share = []
        self.shareModel = []
        self.priors = []

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
        m = iminuit.Minuit(self, self.createParameterList(), name=self.parameter_names)
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
        for source_name in self.pars.keys(): # Loop over every datasource in the created parameters
            p = self.pars[source_name]
            for model_name in p.keys(): # Loop over every model in the datasource
                pars = p[model_name]
                for parameter_name in pars.keys(): # Loop over every parameter in the model
                    parameter = pars[parameter_name]
                    n = '___'.join([source_name, model_name, parameter_name]) # Set a unique name
                    parameter.name = '___'.join([source_name, model_name]) # Set a unique identifiier
                    if parameter_name in self.share: # Set the sharing of a variable with EVERY model
                        if parameter_name in sharing.keys(): # If not the first instance of a shared variable, get the parameter name
                            expr = sharing[parameter_name]
                        else:
                            sharing[parameter_name] = n # If the first instance of a shared variable, set it in the sharing dictionary
                            expr = parameter.expr
                    elif parameter_name in self.shareModel: # Set the sharing of a variable across all models with the SAME NAME
                        if parameter_name in sharingModel.keys() and model_name in sharingModel[parameter_name].keys():
                            expr = sharingModel[parameter_name][model_name]
                        else:
                            try:
                                sharingModel[parameter_name][model_name] = n
                            except:
                                sharingModel[parameter_name] = {model_name: n}
                            expr = parameter.expr
                    else:
                        expr = parameter.expr
                    tuples += ((n, parameter.value, parameter.vary, parameter.min, parameter.max, expr, None),)
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
                        self.mapping[i] = source_name, model_name, parameter_name
                        self.parameter_names.append('_'.join([source_name, model_name, parameter_name]))
                        i += 1
    
    def __call__(self, params):
        for i, value in enumerate(params):
            source_name, model_name, parameter_name = self.mapping[i]
            self.pars[source_name][model_name][parameter_name].value = value
        resid = self.residualCalculation()
        return np.sum(resid*resid)

    def f(self):
        f = []
        for _, source in self.sources:
            f.append(source.f())
        return np.hstack(f)

    @property
    def y(self):
        try:
            return self._y
        except:
            y = []
            for _, source in self.sources:
                y.append(source.y)
            self._y = np.hstack(y)
            return self._y

    @property
    def yerr(self):
        try:
            return self._yerr
        except:
            yerr = []
            for _, source in self.sources:
                yerr.append(source.yerr())
            self._yerr = np.hstack(yerr)
            return self._yerr

    def setParameters(self, params):
        for p in params.keys():
            source_name, model_name, parameter_name = p.split('___')
            self.pars[source_name][model_name][parameter_name].value = params[p].value

    def setUncertainties(self, params):
        for p in params.keys():
            source_name, model_name, parameter_name = p.split('___')
            self.pars[source_name][model_name][parameter_name].unc = params[p].stderr

    def setCorrelations(self, params):
        for p in params.keys():
            source_name, model_name, parameter_name = p.split('___')
            dictionary = copy.deepcopy(params[p].correl)
            del_keys = []
            try:
                keys = list(dictionary.keys())
                for key in keys:
                    if key.startswith(self.pars[source_name][model_name][parameter_name].name):
                        dictionary[key.split('___')[-1]] = dictionary[key]
                    del_keys.append(key)
                for key in del_keys:
                    del dictionary[key]
                self.pars[source_name][model_name][parameter_name].correl = dictionary
            except AttributeError:
                pass

    def resid(self):
        model_calcs = self.f()
        resid = (model_calcs-self.temp_y)/self.yerr()
        if np.any(np.isnan(resid)):
            return np.inf
        else:
            return resid

    def gaussianPriors(self):
        return [(self.pars[source][model][parameter].value - value)/uncertainty for source, model, parameter, value, uncertainty in self.priors]

    def gaussLlh(self):
        resid = self.residualCalculation()
        return -0.5*resid*resid # Faster than **2

    def poissonLlh(self):
        model_calcs = self.f()
        returnvalue = self.temp_y * np.log(model_calcs) - model_calcs
        return returnvalue

    def llh(self, params, method='gaussian', emcee=False):
        methods = {'gaussian': self.gaussLlh, 'poisson': self.poissonLlh}
        self.setParameters(params)
        returnvalue = np.sum(methods[method.lower()]())
        if not np.isfinite(returnvalue):
            returnvalue = -1e99
        if not emcee:
            returnvalue *= -1
        return returnvalue

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

    def fittingDifferenceCalculator(self, parameter_name, llh_selected=False, llh_method='gaussian', method='leastsq', kws={}, mcmc_kwargs={}, sampler_kwargs={}, filename=None):
        if parameter_name not in self.lmpars.keys():
            raise ValueError("Unknown parameter name {}".format(parameter_name))

        fit_kws = {'prepFit': False, 'llh_selected': llh_selected, 'llh_method': llh_method, 'method': method, 'kws': kws, 'mcmc_kwargs': mcmc_kwargs, 'sampler_kwargs': sampler_kwargs, 'filename': filename}
        needed_attr = 'chisqr'
        if llh_selected or llh_method == 'poisson':
            needed_attr = 'nllh_result'
        try:
            original_value = getattr(self, needed_attr)
        except AttributeError:
            self.fit(**fit_kws)
            original_value = getattr(self, needed_attr)

        copied_params = copy.deepcopy(self.pars)
        self.pars[parameter_name].vary = False

        # Define boundary calculating function
        def func(x):
            self.pars[parameter_name].value = x
            self.lmpars = self.pars
            self.fit(**fit_kws)
            value = getattr(self, needed_attr)
            return value - original_value
        return func

    def calculateUncertainties(self, parameter_name, llh_selected=False, llh_method='gaussian', method='leastsq', kws={}, mcmc_kwargs={}, sampler_kwargs={}, filename=None):
        if parameter_name not in self.lmpars.keys():
            raise ValueError("Unknown parameter name {}".format(parameter_name))

        fit_kws = {'prepFit': False, 'llh_selected': llh_selected, 'llh_method': llh_method, 'method': method, 'kws': kws, 'mcmc_kwargs': mcmc_kwargs, 'sampler_kwargs': sampler_kwargs, 'filename': filename}
        diff_calc = self.fittingDifferenceCalculator(parameter_name, **fit_kws)
        if llh_selected or llh_method == 'poisson':
            func_to_zero = lambda x: diff_calc(x) - 0.5
        else:
            func_to_zero = lambda x: diff_calc(x) - 1

    def fit(self, prepFit=True, llh_selected=False, llh_method='gaussian', method='leastsq', kws={}, mcmc_kwargs={}, sampler_kwargs={}, filename=None, steps=1000, nwalkers=50):
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

            kwargs = {'mcmc_kwargs': mcmc_kwargs,
                      'sampler_kwargs': sampler_kwargs}

            kwargs['sampler'] = SATLASSampler
            kwargs['steps'] = steps
            kwargs['nwalkers'] = nwalkers

        self.result = minimize(func, self.lmpars, method=method, iter_cb=self.callback, kws=kws, **kwargs)
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
            self.result = SATLASMinimizer(self.llh, self.lmpars).process_walk(self.lmpars, data)
        except AttributeError:
            self.prepareFit()
            self.result = SATLASMinimizer(self.llh, self.lmpars).process_walk(self.lmpars, data)
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

    # def toDataFrame(self):
    #     import pandas as pd
    #     row = []
    #     df = pd.DataFrame()
    #     for source_name, source in self.sources:
    #         data = {}
    #         row.append(source_name)
    #         p = self.pars[source_name]
    #         d = ()
    #         columns = []
    #         for model_name, model in source.models:
    #             pars = p[model_name]
    #             for parameter_name in pars.keys():
    #                 columns.extend([(model_name, parameter_name, 'Value'), (model_name, parameter_name, 'Uncertainty')])
    #                 d += (pars[parameter_name].value, pars[parameter_name].unc)
    #         columns.extend([(model_name, 'Fit quality', 'Chisquare'), (model_name, 'Fit quality', 'Reduced chisquare'), (model_name, 'Fit quality', 'NDoF')])
    #         d += (source.chisqr, source.redchi, source.nfree)
    #         data[source_name] = d
    #         d = pd.DataFrame.from_dict(data, orient='index')
    #         d.columns = pd.MultiIndex.from_tuples(columns)
    #         df = pd.concat([df, d])
    #     df.sort_index(axis=1, level=1, ascending=True, sort_remaining=False, inplace=True)
    #     return df

class Fitter:
    def __init__(self):
        super().__init__()
        self.sources = []
        self.pars = {}
        self.bounds = optimize.Bounds([], [])
        self.share = []
        self.shareModel = []
        self.priors = []

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
        m = iminuit.Minuit(self, self.createParameterList(), name=self.parameter_names)
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
        for source_name in self.pars.keys(): # Loop over every datasource in the created parameters
            p = self.pars[source_name]
            for model_name in p.keys(): # Loop over every model in the datasource
                pars = p[model_name]
                for parameter_name in pars.keys(): # Loop over every parameter in the model
                    parameter = pars[parameter_name]
                    n = '___'.join([source_name, model_name, parameter_name]) # Set a unique name
                    parameter.name = '___'.join([source_name, model_name]) # Set a unique identifiier
                    if parameter_name in self.share: # Set the sharing of a variable with EVERY model
                        if parameter_name in sharing.keys(): # If not the first instance of a shared variable, get the parameter name
                            expr = sharing[parameter_name]
                        else:
                            sharing[parameter_name] = n # If the first instance of a shared variable, set it in the sharing dictionary
                            expr = parameter.expr
                    elif parameter_name in self.shareModel: # Set the sharing of a variable across all models with the SAME NAME
                        if parameter_name in sharingModel.keys() and model_name in sharingModel[parameter_name].keys():
                            expr = sharingModel[parameter_name][model_name]
                        else:
                            try:
                                sharingModel[parameter_name][model_name] = n
                            except:
                                sharingModel[parameter_name] = {model_name: n}
                            expr = parameter.expr
                    else:
                        expr = parameter.expr
                    tuples += ((n, parameter.value, parameter.vary, parameter.min, parameter.max, expr, None),)
        lmpars.add_many(*tuples)
        self.lmpars = lmpars
    
    def setExpr(self, name, expr):
        self.createLmParameters()
        if name in self.lmpars.keys():
            self.lmpars[name].expr = expr

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
                        self.mapping[i] = source_name, model_name, parameter_name
                        self.parameter_names.append('_'.join([source_name, model_name, parameter_name]))
                        i += 1
    
    def __call__(self, params):
        for i, value in enumerate(params):
            source_name, model_name, parameter_name = self.mapping[i]
            self.pars[source_name][model_name][parameter_name].value = value
        resid = self.residualCalculation()
        return np.sum(resid*resid)

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
            self.pars[source_name][model_name][parameter_name].value = params[p].value

    def setUncertainties(self, params):
        for p in params.keys():
            source_name, model_name, parameter_name = p.split('___')
            self.pars[source_name][model_name][parameter_name].unc = params[p].stderr

    def setCorrelations(self, params):
        for p in params.keys():
            source_name, model_name, parameter_name = p.split('___')
            dictionary = copy.deepcopy(params[p].correl)
            del_keys = []
            try:
                keys = list(dictionary.keys())
                for key in keys:
                    if key.startswith(self.pars[source_name][model_name][parameter_name].name):
                        dictionary[key.split('___')[-1]] = dictionary[key]
                    del_keys.append(key)
                for key in del_keys:
                    del dictionary[key]
                self.pars[source_name][model_name][parameter_name].correl = dictionary
            except AttributeError:
                pass

    def resid(self):
        model_calcs = self.f()
        resid = (model_calcs-self.temp_y)/self.yerr()
        if np.any(np.isnan(resid)):
            return np.inf
        else:
            return resid

    def gaussianPriors(self):
        return [(self.pars[source][model][parameter].value - value)/uncertainty for source, model, parameter, value, uncertainty in self.priors]

    def gaussLlh(self):
        resid = self.residualCalculation()
        # print('here')
        # raise ImportError
        # print('here')
        return 0.5*resid*resid # Faster than **2

    def poissonLlh(self):
        model_calcs = self.f()
        returnvalue = self.temp_y * np.log(model_calcs) - model_calcs
        return -returnvalue

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

    def fit(self, prepFit=True, llh_selected=False, llh_method='gaussian', method='leastsq', kws={}, mcmc_kwargs={}, sampler_kwargs={}, filename=None, steps=1000, nwalkers=50):
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

            kwargs = {'mcmc_kwargs': mcmc_kwargs,
                      'sampler_kwargs': sampler_kwargs}

            kwargs['sampler'] = SATLASSampler
            kwargs['steps'] = steps
            kwargs['nwalkers'] = nwalkers

        reduce_fcn = self.reduction
        self.result = minimize(func, self.lmpars, method=method, iter_cb=self.callback, kws=kws, reduce_fcn=reduce_fcn, **kwargs)
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
            self.result = SATLASMinimizer(self.llh, self.lmpars).process_walk(self.lmpars, data)
        except AttributeError:
            self.prepareFit()
            self.result = SATLASMinimizer(self.llh, self.lmpars).process_walk(self.lmpars, data)
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
    def __init__(self, x, y, xerr=None, yerr=1, name=None):
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
        for name, model in self.models:
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
        if not callable(self.yerr_data):
            return self.yerr_data
        else:
            return self.yerr_data(self.f())

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
        return '{}+/-{} ({} max, {} min, vary={}, correl={})'.format(self.value, self.unc, self.max, self.min, self.vary, self.correl)
