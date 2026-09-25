"""
Reimplementation of several features in the emcee and lmfit packages, in order to make it work correctly.

The random walk (:meth:`SATLASMinimizer.emcee`) follows :meth:`lmfit.Minimizer.emcee`,
with these differences:

* the walk can be saved to, and resumed from, an HDF5 file
  (:class:`SATLASHDFBackend`, which also stores the parameter names);
* the walk can stop when the autocorrelation time has converged;
* keyword arguments can be passed to the sampler and to its ``sample`` method.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""

import logging
import multiprocessing
import warnings
from typing import Dict, List, Optional, Union

import emcee
import lmfit.minimizer
import numpy as np
from emcee.autocorr import AutocorrError
from lmfit import Minimizer

try:
    import dill  # noqa: F401

    HAS_DILL = True
except ImportError:
    HAS_DILL = False

AbortFitException = lmfit.minimizer.AbortFitException
_make_random_gen = lmfit.minimizer._make_random_gen
try:
    _coerce_float64 = lmfit.minimizer.coerce_float64
except AttributeError:  # lmfit < 1.3
    _coerce_float64 = lmfit.minimizer._nan_policy

__all__ = ["SATLASSampler", "SATLASHDFBackend", "SATLASMinimizer", "minimize"]

logger = logging.getLogger(__name__)

# Relative spread of the walkers around the starting values
_INITIAL_SPREAD = 1.0e-4


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
    """Ensemble sampler that accepts log-probabilities returned as arrays of
    one element, as the likelihood of the :class:`~satlas2.core.Fitter` does."""

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
            map_func = self.pool.map if self.pool is not None else map
            results = map_func(self.log_prob_fn, p)

        # lmfit returns an array of one element for walkers inside the bounds,
        # and a float (-inf) for walkers outside of them
        log_prob = np.array([np.asarray(l, dtype=float).item() for l in results])

        # Check for log_prob returning NaN.
        if np.any(np.isnan(log_prob)):
            raise ValueError("Probability function returned NaN")

        return log_prob, None


class SATLASHDFBackend(emcee.backends.HDFBackend):
    """HDF5 backend that also stores the names of the parameters."""

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


def _autocorrelationTime(get_time) -> np.ndarray:
    """Integrated autocorrelation time; if the chain is too short for a
    reliable estimate, warn and return the unreliable estimate."""
    try:
        return get_time()
    except AutocorrError as e:
        warnings.warn(str(e), RuntimeWarning, stacklevel=3)
        return get_time(tol=0)


class SATLASMinimizer(Minimizer):
    def process_walk(self, params, chain):
        """Summarise a random walk read from a file, as :meth:`emcee` does
        for a walk it has just performed."""
        result = self.prepare_fit(params)
        result.method = "emcee"
        steps, nwalkers = chain.shape[:2]
        self._summariseChain(result, chain.reshape((-1, result.nvarys)))
        result.errorbars = True
        result.nvarys = len(result.var_names)
        result.nfev = nwalkers * steps
        result.acor = _autocorrelationTime(
            lambda **kw: emcee.autocorr.integrated_time(chain, **kw)
        )
        result.message = "MCMC walk processed successfully."
        result.success = True
        return result

    @staticmethod
    def _summariseChain(result, flatchain: np.ndarray) -> None:
        """Set the parameters to the median of the walk, with half the 68%
        interval as uncertainty and the correlations between the parameters."""
        params = result.params
        quantiles = np.percentile(flatchain, [15.87, 50, 84.13], axis=0)
        for i, var_name in enumerate(result.var_names):
            std_l, median, std_u = quantiles[:, i]
            params[var_name].value = median
            params[var_name].stderr = 0.5 * (std_u - std_l)
            params[var_name].correl = {}
        params.update_constraints()

        corrcoefs = np.corrcoef(flatchain.T)
        for i, var_name in enumerate(result.var_names):
            for j, var_name2 in enumerate(result.var_names):
                if i != j:
                    params[var_name].correl[var_name2] = corrcoefs[i, j]

    @staticmethod
    def _walkBounds(params, var_names):
        """Starting values and bounds of the varied parameters, without the
        internal parameter scaling of lmfit."""
        values, bounds = [], []
        for param in params.values():
            if param.expr is not None:
                param.vary = False
            if not param.vary:
                continue
            values.append(param.value)
            param.from_internal = lambda val: val
            lb = -np.inf if param.min is None or param.min is np.nan else param.min
            ub = np.inf if param.max is None or param.max is np.nan else param.max
            bounds.append((lb, ub))
        return np.array(values, dtype=float).reshape(len(var_names)), np.array(bounds)

    def _prepareBackend(self, backend, load: bool, nwalkers: int, var_names) -> int:
        """Reset the backend for a new walk, or read the number of walkers
        when resuming one. Returns the number of walkers."""
        if backend is None:
            return nwalkers
        if load:
            nwalkers = backend.shape[0]
        else:
            backend.reset(nwalkers, self.nvarys)
        backend.labels = var_names
        return nwalkers

    def _sample(
        self,
        p0,
        steps: int,
        progress: bool,
        mcmc_kwargs: dict,
        convergence: bool,
        convergence_iter: int,
        convergence_tau: float,
    ):
        """Run the sampler, optionally stopping when the autocorrelation time
        is shorter than 1/convergence_iter of the walk and changed less than
        convergence_tau (relatively) since the previous check.

        Returns the last position of the walkers and whether the walk converged."""
        if p0 is None:
            p0 = self.sampler._previous_state
        check_every = int(np.ceil(1000 / self.sampler.nwalkers))
        old_tau = np.inf
        output = None
        converged = False
        for output in self.sampler.sample(
            p0, iterations=steps, progress=progress, **mcmc_kwargs
        ):
            if not convergence or self.sampler.iteration % check_every:
                continue
            tau = self.sampler.get_autocorr_time(tol=0)
            converged = np.all(tau * convergence_iter < self.sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < convergence_tau)
            if converged:
                logger.info("emcee stopped due to convergence")
                break
            old_tau = tau
        return output.coords, converged

    def _startPosition(self, p0, pos, reuse_sampler: bool):
        """Starting position given by the user, if any."""
        if pos is None or reuse_sampler:
            return p0
        tpos = np.asarray(pos, dtype=float)
        if p0 is not None and p0.shape == tpos.shape:
            return tpos
        # trying to initialise with a previous chain
        if tpos.shape[-1] == self.nvarys:
            return tpos[-1]
        raise ValueError("pos should have shape (nwalkers, nvarys)")

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
        mcmc_kwargs=None,
        sampler_kwargs=None,
        sampler=emcee.EnsembleSampler,
    ):
        """Perform a random walk. See :meth:`lmfit.Minimizer.emcee` for the
        common arguments; the others are described in the module documentation
        and in :meth:`satlas2.core.Fitter.fit`."""
        if ntemps > 1:
            raise DeprecationWarning(
                "'ntemps' has no effect anymore, since the PTSampler was "
                "removed from emcee version 3."
            )
        mcmc_kwargs = dict(mcmc_kwargs or {})
        sampler_kwargs = dict(sampler_kwargs or {})

        tparams = params
        # if you're reusing the sampler then nwalkers have to be
        # determined from the previous sampling
        if reuse_sampler:
            if not hasattr(self, "sampler") or not hasattr(self, "_lastpos"):
                raise ValueError(
                    "You wanted to use an existing sampler, but "
                    "it hasn't been created yet"
                )
            nwalkers = self._lastpos.shape[-2]
            tparams = None

        result = self.prepare_fit(params=tparams)
        params = result.params

        # check if the userfcn returns a vector of residuals
        out = np.asarray(self.userfcn(params, *self.userargs, **self.userkws)).ravel()
        if out.size > 1 and is_weighted is False and "__lnsigma" not in params:
            # marginalise over a constant data uncertainty
            params.add("__lnsigma", value=0.01, min=-np.inf, max=np.inf, vary=True)
            result = self.prepare_fit(params)
            params = result.params

        result.method = "emcee"
        var_arr, bounds = self._walkBounds(params, result.var_names)
        self.nvarys = len(result.var_names)

        # set up multiprocessing; a pool in sampler_kwargs is used as given
        auto_pool = None
        if isinstance(workers, int) and workers > 1 and HAS_DILL:
            auto_pool = multiprocessing.Pool(workers)
            sampler_kwargs["pool"] = auto_pool
        elif hasattr(workers, "map"):
            sampler_kwargs["pool"] = workers

        # arguments sent to the log-probability function by the sampler
        sampler_kwargs["args"] = (self.userfcn, params, result.var_names, bounds)
        sampler_kwargs["kwargs"] = {
            "is_weighted": is_weighted,
            "float_behavior": float_behavior,
            "userargs": self.userargs,
            "userkws": self.userkws,
            "nan_policy": self.nan_policy,
        }

        rng = _make_random_gen(seed)
        backend = sampler_kwargs.get("backend")
        nwalkers = self._prepareBackend(backend, load, nwalkers, result.var_names)
        p0 = None
        if not load:
            p0 = (1 + rng.randn(nwalkers, self.nvarys) * _INITIAL_SPREAD) * var_arr
        self.sampler = sampler(nwalkers, self.nvarys, self._lnprob, **sampler_kwargs)
        p0 = self._startPosition(p0, pos, reuse_sampler)
        # if you specified a seed then you also need to seed the sampler
        if seed is not None:
            self.sampler.random_state = rng.get_state()

        converged = False
        try:
            self._lastpos, converged = self._sample(
                p0,
                steps,
                progress,
                mcmc_kwargs,
                convergence,
                convergence_iter,
                convergence_tau,
            )
        except AbortFitException:
            result.aborted = True
            result.message = "Fit aborted by user callback. Could not estimate error-bars."
            result.success = False

        # discard the burn samples and thin
        chain = self.sampler.get_chain(thin=thin, discard=burn)
        if not result.aborted:
            self._summariseChain(result, chain.reshape((-1, self.nvarys)))
        result.chain = np.copy(chain)
        result.lnprob = np.copy(self.sampler.get_log_prob(thin=thin, discard=burn))
        result.errorbars = True
        result.nvarys = len(result.var_names)
        result.nfev = nwalkers * steps
        result.acor = _autocorrelationTime(self.sampler.get_autocorr_time)
        result.acceptance_fraction = self.sampler.acceptance_fraction

        self._calculateWalkStatistics(result, params, is_weighted, float_behavior)

        if auto_pool is not None:
            auto_pool.terminate()
        if not result.aborted:
            result.message = (
                "MCMC sampling stopped early: the autocorrelation time converged."
                if converged
                else "MCMC sampling completed successfully."
            )
            result.success = True
        return result

    def _calculateWalkStatistics(
        self, result, params, is_weighted: bool, float_behavior: str
    ) -> None:
        """Residual and fit statistics at the median of the walk."""
        out = self.userfcn(params, *self.userargs, **self.userkws)
        result.residual = _coerce_float64(
            out, nan_policy=self.nan_policy, handle_inf=False
        )

        # If uncertainty was automatically estimated, weight the residual properly
        if (not is_weighted) and result.residual.size > 1 and "__lnsigma" in params:
            result.residual = result.residual / np.exp(params["__lnsigma"].value)

        if isinstance(result.residual, np.ndarray) or float_behavior == "chi2":
            result._calculate_statistics()
        elif float_behavior == "posterior":
            # special case unique to emcee
            result.ndata = 1
            result.nfree = 1
            # assuming prior prob = 1, this is true
            _neg2_log_likel = -2 * result.residual
            # assumes that residual is properly weighted, avoid overflowing np.exp()
            result.chisqr = np.exp(min(650, _neg2_log_likel))
            result.redchi = result.chisqr / result.nfree
            result.aic = _neg2_log_likel + 2 * result.nvarys
            result.bic = _neg2_log_likel + np.log(result.ndata) * result.nvarys


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
