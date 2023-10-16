"""
Reimplementation of several features in the emcee and lmfit packages, in order to make it work correctly.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""
import multiprocessing

import emcee
import lmfit.minimizer
import numpy as np
from emcee.autocorr import AutocorrError
from lmfit import Minimizer

AbortFitException = lmfit.minimizer.AbortFitException
from typing import Dict, List, Union

try:
    import pandas as pd
    from pandas import isnull

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    isnull = np.isnan
try:
    import dill  # noqa: F401

    HAS_DILL = True
except ImportError:
    HAS_DILL = False
_make_random_gen = lmfit.minimizer._make_random_gen
isnull = lmfit.minimizer.isnull
_nan_policy = lmfit.minimizer._nan_policy

__all__ = ["SATLASSampler", "SATLASHDFBackend", "SATLASMinimizer", "minimize"]


def ndarray_to_list_of_dicts(
    x: np.ndarray, key_map: Dict[str, Union[int, List[int]]]
) -> List[Dict[str, Union[np.number, np.ndarray]]]:
    """
    A helper function to convert a ``np.ndarray`` into a list
    of dictionaries of parameters. Used when parameters are named.
    Args:
      x (np.ndarray): parameter array of shape ``(N, n_dim)``, where
        ``N`` is an integer
      key_map (Dict[str, Union[int, List[int]]):
    Returns:
      list of dictionaries of parameters
    """
    return [{key: xi[val] for key, val in key_map.items()} for xi in x]


class SATLASSampler(emcee.EnsembleSampler):
    def compute_log_prob(self, coords):
        """Calculate the vector of log-probability for the walkers

        Args:
            coords: (ndarray[..., ndim]) The position vector in parameter
                space where the probability should be calculated.

        This method returns:

        * log_prob: A vector of log-probabilities with one entry for each
          walker in this sub-ensemble.
        * blob: The list of meta data returned by the ``log_post_fn`` at
          this position or ``None`` if nothing was returned.

        """
        p = coords

        # Check that the parameters are in physical ranges.
        if np.any(np.isinf(p)):
            raise ValueError("At least one parameter value was infinite")
        if np.any(np.isnan(p)):
            raise ValueError("At least one parameter value was NaN")

        # If the parmaeters are named, then switch to dictionaries
        if self.params_are_named:
            p = ndarray_to_list_of_dicts(p, self.parameter_names)

        # Run the log-probability calculations (optionally in parallel).
        if self.vectorize:
            results = self.log_prob_fn(p)
        else:
            # If the `pool` property of the sampler has been set (i.e. we want
            # to use `multiprocessing`), use the `pool`'s map method.
            # Otherwise, just use the built-in `map` function.
            if self.pool is not None:
                map_func = self.pool.map
            else:
                map_func = map
            results = list(map_func(self.log_prob_fn, p))

        log_prob = np.array([float(l) for l in results])
        blob = None

        # Check for log_prob returning NaN.
        if np.any(np.isnan(log_prob)):
            raise ValueError("Probability function returned NaN")

        return log_prob, blob


class SATLASHDFBackend(emcee.backends.HDFBackend):
    @property
    def labels(self):
        with self.open() as f:
            g = f[self.name]
            return g.attrs["labels"]

    @labels.setter
    def labels(self, labels):
        with self.open("a") as f:
            g = f[self.name]
            g.attrs["labels"] = labels


class SATLASMinimizer(Minimizer):
    def process_walk(self, params, chain):
        result = self.prepare_fit(params)
        params = result.params
        nvarys = result.nvarys
        result.method = "emcee"

        flatchain = chain.reshape((-1, nvarys))
        steps = chain.shape[0]
        nwalkers = chain.shape[1]
        quantiles = np.percentile(flatchain, [15.87, 50, 84.13], axis=0)

        for i, var_name in enumerate(result.var_names):
            std_l, median, std_u = quantiles[:, i]
            params[var_name].value = median
            params[var_name].stderr = 0.5 * (std_u - std_l)
            params[var_name].correl = {}

        params.update_constraints()

        # work out correlation coefficients
        corrcoefs = np.corrcoef(flatchain.T)

        for i, var_name in enumerate(result.var_names):
            for j, var_name2 in enumerate(result.var_names):
                if i != j:
                    result.params[var_name].correl[var_name2] = corrcoefs[i, j]

        result.errorbars = True
        result.nvarys = len(result.var_names)
        result.nfev = nwalkers * steps

        try:
            result.acor = emcee.autocorr.integrated_time(chain)
        except AutocorrError as e:
            print(str(e))
            result.acor = emcee.autocorr.integrated_time(chain, tol=0)
        return result

        # Calculate the residual with the "best fit" parameters

    def emcee(
        self,
        params=None,
        steps=1000,
        nwalkers=100,
        burn=0,
        thin=1,
        ntemps=1,
        load=False,
        convergence=False,
        convergence_iter=50,
        convergence_tau=0.01,
        pos=None,
        reuse_sampler=False,
        workers=1,
        float_behavior="posterior",
        is_weighted=True,
        seed=None,
        progress=True,
        mcmc_kwargs={},
        sampler_kwargs={},
        sampler=emcee.EnsembleSampler,
    ):
        if ntemps > 1:
            msg = (
                "'ntemps' has no effect anymore, since the PTSampler was "
                "removed from emcee version 3."
            )
            raise DeprecationWarning(msg)

        tparams = params
        # if you're reusing the sampler then nwalkers have to be
        # determined from the previous sampling
        if reuse_sampler:
            if not hasattr(self, "sampler") or not hasattr(self, "_lastpos"):
                raise ValueError(
                    "You wanted to use an existing sampler, but "
                    "it hasn't been created yet"
                )
            if len(self._lastpos.shape) == 2:
                nwalkers = self._lastpos.shape[0]
            elif len(self._lastpos.shape) == 3:
                nwalkers = self._lastpos.shape[1]
            tparams = None

        result = self.prepare_fit(params=tparams)
        params = result.params

        # check if the userfcn returns a vector of residuals
        out = self.userfcn(params, *self.userargs, **self.userkws)
        out = np.asarray(out).ravel()
        if out.size > 1 and is_weighted is False:
            # we need to marginalise over a constant data uncertainty
            if "__lnsigma" not in params:
                # __lnsigma should already be in params if is_weighted was
                # previously set to True.
                params.add(
                    "__lnsigma", value=0.01, min=-np.inf, max=np.inf, vary=True
                )
                # have to re-prepare the fit
                result = self.prepare_fit(params)
                params = result.params

        result.method = "emcee"

        # Removing internal parameter scaling. We could possibly keep it,
        # but I don't know how this affects the emcee sampling.
        bounds = []
        var_arr = np.zeros(len(result.var_names))
        i = 0
        for par in params:
            param = params[par]
            if param.expr is not None:
                param.vary = False
            if param.vary:
                var_arr[i] = param.value
                i += 1
            else:
                # don't want to append bounds if they're not being varied.
                continue

            param.from_internal = lambda val: val
            lb, ub = param.min, param.max
            if lb is None or lb is np.nan:
                lb = -np.inf
            if ub is None or ub is np.nan:
                ub = np.inf
            bounds.append((lb, ub))
        bounds = np.array(bounds)

        self.nvarys = len(result.var_names)

        # set up multiprocessing options for the samplers
        auto_pool = None
        # sampler_kwargs = {}
        if isinstance(workers, int) and workers > 1 and HAS_DILL:
            auto_pool = multiprocessing.Pool(workers)
            sampler_kwargs["pool"] = auto_pool
        elif hasattr(workers, "map"):
            sampler_kwargs["pool"] = workers

        # function arguments for the log-probability functions
        # these values are sent to the log-probability functions by the sampler.
        lnprob_args = (self.userfcn, params, result.var_names, bounds)
        lnprob_kwargs = {
            "is_weighted": is_weighted,
            "float_behavior": float_behavior,
            "userargs": self.userargs,
            "userkws": self.userkws,
            "nan_policy": self.nan_policy,
        }

        sampler_kwargs["args"] = lnprob_args
        sampler_kwargs["kwargs"] = lnprob_kwargs

        # set up the random number generator
        rng = _make_random_gen(seed)

        backend = sampler_kwargs.pop("backend")
        if backend is not None:
            if not load:
                backend.reset(nwalkers, self.nvarys)
            else:
                nwalkers = backend.shape[0]
            backend.labels = result.var_names
        sampler_kwargs["backend"] = backend
        # now initialise the samplers

        if load:
            p0 = None
        else:
            p0 = 1 + rng.randn(nwalkers, self.nvarys) * 1.0e-4
            p0 *= var_arr
        sampler_kwargs["pool"] = auto_pool
        self.sampler = sampler(
            nwalkers, self.nvarys, self._lnprob, **sampler_kwargs
        )

        # user supplies an initialisation position for the chain
        # If you try to run the sampler with p0 of a wrong size then you'll get
        # a ValueError. Note, you can't initialise with a position if you are
        # reusing the sampler.
        if pos is not None and not reuse_sampler:
            tpos = np.asfarray(pos)
            if p0.shape == tpos.shape:
                pass
            # trying to initialise with a previous chain
            elif tpos.shape[-1] == self.nvarys:
                tpos = tpos[-1]
            else:
                raise ValueError("pos should have shape (nwalkers, nvarys)")
            p0 = tpos

        # if you specified a seed then you also need to seed the sampler
        if seed is not None:
            self.sampler.random_state = rng.get_state()

        # now do a production run, sampling all the time
        try:
            output = None
            old_tau = np.inf
            check = int(np.ceil(1000 / nwalkers))
            converged = False
            if p0 is None:
                p0 = self.sampler._previous_state
            for output in self.sampler.sample(
                p0, iterations=steps, progress=progress, **mcmc_kwargs
            ):
                if convergence:
                    if self.sampler.iteration % check:
                        continue
                    tau = self.sampler.get_autocorr_time(tol=0)
                    converged = np.all(
                        tau * convergence_iter < self.sampler.iteration
                    )
                    converged &= np.all(
                        np.abs(old_tau - tau) / tau < convergence_tau
                    )
                    if converged:
                        break
                    old_tau = tau
            if converged:
                print("emcee stopped due to convergence")
            self._lastpos = output.coords
        except AbortFitException:
            result.aborted = True
            result.message = (
                "Fit aborted by user callback. Could not estimate error-bars."
            )
            result.success = False
            result.nfev = self.result.nfev
            output = None

        # discard the burn samples and thin
        chain = self.sampler.get_chain(thin=thin, discard=burn)[..., :, :]
        lnprobability = self.sampler.get_log_prob(thin=thin, discard=burn)[
            ..., :
        ]
        flatchain = chain.reshape((-1, self.nvarys))
        if not result.aborted:
            quantiles = np.percentile(flatchain, [15.87, 50, 84.13], axis=0)

            for i, var_name in enumerate(result.var_names):
                std_l, median, std_u = quantiles[:, i]
                params[var_name].value = median
                params[var_name].stderr = 0.5 * (std_u - std_l)
                params[var_name].correl = {}

            params.update_constraints()

            # work out correlation coefficients
            corrcoefs = np.corrcoef(flatchain.T)

            for i, var_name in enumerate(result.var_names):
                for j, var_name2 in enumerate(result.var_names):
                    if i != j:
                        result.params[var_name].correl[var_name2] = corrcoefs[
                            i, j
                        ]

        result.chain = np.copy(chain)
        result.lnprob = np.copy(lnprobability)
        result.errorbars = True
        result.nvarys = len(result.var_names)
        result.nfev = nwalkers * steps

        try:
            result.acor = self.sampler.get_autocorr_time()
        except AutocorrError as e:
            print(str(e))
            result.acor = self.sampler.get_autocorr_time(tol=0, quiet=True)
        result.acceptance_fraction = self.sampler.acceptance_fraction

        # Calculate the residual with the "best fit" parameters
        out = self.userfcn(params, *self.userargs, **self.userkws)
        result.residual = _nan_policy(
            out, nan_policy=self.nan_policy, handle_inf=False
        )

        # If uncertainty was automatically estimated, weight the residual properly
        if (not is_weighted) and (result.residual.size > 1):
            if "__lnsigma" in params:
                result.residual = result.residual / np.exp(
                    params["__lnsigma"].value
                )

        # Calculate statistics for the two standard cases:
        if isinstance(result.residual, np.ndarray) or (
            float_behavior == "chi2"
        ):
            result._calculate_statistics()

        # Handle special case unique to emcee:
        # This should eventually be moved into result._calculate_statistics.
        elif float_behavior == "posterior":
            result.ndata = 1
            result.nfree = 1

            # assuming prior prob = 1, this is true
            _neg2_log_likel = -2 * result.residual

            # assumes that residual is properly weighted, avoid overflowing np.exp()
            result.chisqr = np.exp(min(650, _neg2_log_likel))

            result.redchi = result.chisqr / result.nfree
            result.aic = _neg2_log_likel + 2 * result.nvarys
            result.bic = _neg2_log_likel + np.log(result.ndata) * result.nvarys

        if auto_pool is not None:
            auto_pool.terminate()

        return result


def minimize(
    fcn,
    params,
    method="leastsq",
    args=None,
    kws=None,
    iter_cb=None,
    scale_covar=True,
    nan_policy="raise",
    reduce_fcn=None,
    calc_covar=True,
    max_nfev=None,
    **fit_kws,
):
    minimizer_kws = fit_kws.pop("minimizer_kws", {})
    fitter = SATLASMinimizer(
        fcn,
        params,
        fcn_args=args,
        fcn_kws=kws,
        iter_cb=iter_cb,
        scale_covar=scale_covar,
        nan_policy=nan_policy,
        reduce_fcn=reduce_fcn,
        calc_covar=calc_covar,
        max_nfev=max_nfev,
        **fit_kws,
    )
    return fitter.minimize(method=method, **minimizer_kws)
