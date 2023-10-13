"""
Implementation of the base HFSModel and SumModel classes, based on the syntax used in the original satlas

NOTE: THIS IS NOT FULLY BENCHMARKED/DEVELOPED SO BUGS MIGHT BE PRESENT, AND NOT ALL FUNCTIONALITIES OF THE ORIGINAL SATLAS ARE IMPLEMENTED

.. moduleauthor:: Bram van den Borne <bram.vandenborne@kuleuven.be>
"""

from __future__ import annotations

from typing import Tuple, Union

import lmfit as lm
import pandas as pd
from numpy.typing import ArrayLike

from .core import Fitter, Source
from .models import HFS, PiecewiseConstant, Polynomial


class HFSModel:
    """Initializes a hyperfine spectrum Model with the given hyperfine parameters.

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
    """

    def __init__(
        self,
        I: float,
        J: ArrayLike[float, float],
        ABC: ArrayLike[float, float, float, float, float, float],
        centroid: float = 0,
        fwhm: ArrayLike[float, float] = [50.0, 50.0],
        scale: float = 1.0,
        background_params: ArrayLike = [0.001],
        shape: str = "voigt",
        use_racah: bool = True,
        use_saturation: bool = False,
        saturation: float = 0.001,
        sidepeak_params: dict = {"N": None, "Poisson": 0, "Offset": 0},
        crystalballparams=None,
        pseudovoigtparams=None,
        asymmetryparams=None,
        name: str = "HFModel__",
    ):
        super(HFSModel, self).__init__()
        self.background_params = background_params
        if shape != "voigt":
            raise NotImplementedError("Only Voigt shape is supported.")
        if (
            crystalballparams != None
            or pseudovoigtparams != None
            or asymmetryparams != None
        ):
            raise NotImplementedError("Only Voigt shape is supported.")
        if name == "HFModel__":
            self.name = name + str(I).replace(".", "_")
        self.hfs = HFS(
            I,
            J,
            A=ABC[:2],
            B=ABC[2:4],
            C=ABC[4:6],
            scale=scale,
            df=centroid,
            fwhmg=fwhm[0],
            fwhml=fwhm[1],
            name=self.name,
            racah=use_racah,
            N=sidepeak_params["N"],
            offset=sidepeak_params["Offset"],
            poisson=sidepeak_params["Poisson"],
            prefunc=None,
        )
        self.params = self.hfs.params

    def set_expr(self, constraints: dict) -> None:
        """Set the expression to be used for the given parameters.
        The constraint should be a dict with following structure:

        key: string
            Parameter to constrain
        value: ArrayLike, length = 2
            First element: Factor to multiply
            Second element: Parameter that the key should be constrained to. {'Au':['0.5','Al']} results in Au = 0.5*Al
        """
        for cons in constraints.keys():
            self.hfs.params[
                cons
            ].expr = f"{constraints[cons][0]}*Fit___{self.name}___{constraints[cons][1]}"

    def fix_ratio(self, value, target="upper", parameter="A"):
        raise NotImplementedError("Use HFSModel.set_expr(...)")

    def set_variation(self, varyDict: dict) -> None:
        """Sets the variation of the fitparameters as supplied in the
        dictionary.

        Parameters
        ----------
        varyDict: dictionary
            A dictionary containing 'key: True/False' mappings with
            the parameter names as keys."""

        for p in varyDict.keys():
            self.hfs.params[p].vary = False

    def f(self, x: ArrayLike) -> ArrayLike:
        """Calculate the response for an unshifted spectrum with background

        Parameters
        ----------
        x : ArrayLike

        Returns
        -------
        ArrayLike
        """
        return self.hfs.fUnshifted(x) + Polynomial(
            self.background_params, name="bkg"
        ).f(x)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Calculate the response for an unshifted spectrum with background

        Parameters
        ----------
        x : ArrayLike

        Returns
        -------
        ArrayLike
        """
        return self.hfs.fUnshifted(x) + Polynomial(
            self.background_params, name="bkg"
        ).f(x)

    def chisquare_fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        yerr: Union[ArrayLike, callable] = None,
        xerr: ArrayLike = None,
        func: callable = None,
        verbose: bool = False,
        hessian: bool = False,
        method: str = "leastsq",
        show_correl: bool = True,
    ) -> Tuple[bool, str]:
        """Perform a fit of this model to the data provided in this function.

        Parameters
        ----------
        x : ArrayLike
            x-values of the data points
        y : ArrayLike
            y-values of the data-points
        yerr : Union[ArrayLike, callable], optional
            1-sigma error on the y-values, values or function to be used on f(x),
            by default None=sqrt(f(x))
        xerr : ArrayLike, optional
            1-sigma error on the x-values, by default None
        func : callable, optional
            Not implemented
        verbose : bool, optional
            Not implemented, by default False
        hessian : bool, optional
            Not implemented, by default False
        method : str, optional
            Selects the method used by the :func:`lmfit.minimizer`, by default 'leastsq'

        Returns
        -------
        Tuple[bool, str]
            Returns a boolean indicating the success of the fit, and the message accompanying it.

        Raises
        ------
        NotImplementedError
            When the chosen options for func, verbose and Hessian result is not implemented.
        """
        if show_correl:
            print(
                "define whether you want to see the correlations in display_chisquare_fit(...)"
            )
        if func is not None:
            yerr = func
        datasource = Source(x, y, yerr=yerr, name="Fit")
        datasource.addModel(self.hfs)
        bkg = Polynomial(self.background_params, name="bkg")
        datasource.addModel(bkg)
        self.fitter = Fitter()
        self.fitter.addSource(datasource)
        self.fitter.fit(method=method)
        self.background_params = [
            list(bkg.params.values())[i].value
            for i in range(len(list(bkg.params.values())))
        ]
        return self.fitter.result.success, self.fitter.result.message

    def display_chisquare_fit(self, scaled: bool = True, **kwargs):
        """Generate a report of the fitting results.

        The report contains the best-fit values for the parameters and their uncertainties and correlations.

        Parameters
        ----------
        scaled: bool, optional
            Whether the errors are scaled with reduced chisquared, by default True, and only True
        show_correl : bool, optional
            Whether to show a list of sorted correlations, by default False
        min_correl : float, optional
            Smallest correlation in absolute value to show, by default 0.1

        Returns
        -------
        str
            Multi-line text of fit report.
        """
        if not scaled:
            raise NotImplementedError("Not implemented")
        return lm.fit_report(self.fitter.result, **kwargs)

    def get_result(
        self, selection: str = "chisquare"
    ) -> Tuple[list, list, list]:
        """Return the variable names, values and estimated error bars for the
        parameters as seperate lists.

        Parameters
        ----------
        selection: string, optional
            Selects if the chisquare ('chisquare' or 'any') or MLE values are
            used. Defaults to 'chisquare', and chisquare only

        Returns
        -------
        names, values, uncertainties: tuple of lists
            Returns a 3-tuple of lists containing the names of the parameters,
            the values and the estimated uncertainties, scaled with the reduced chisquared.
        """
        lmparamdict = self.fitter.pars["Fit"][self.name]
        return (
            list(lmparamdict.keys()),
            [
                lmparamdict[param_name].value
                for param_name in lmparamdict.keys()
            ],
            [lmparamdict[param_name].unc for param_name in lmparamdict.keys()],
        )

    def get_result_dict(
        self, method: str = "chisquare", scaled: bool = True
    ) -> dict:
        """Returns the fitted parameters in a dictionary of the form {name: [value, uncertainty]}.

        Parameters
        ----------
        method: {'chisquare', 'mle'}
            Selects which parameters have to be returned, by default 'chisquare', and only 'chisquare'
        scaled: boolean
            Selects if, in case of chisquare parameters, the uncertainty
            has to be scaled by sqrt(reduced_chisquare). Defaults to True, and only True

        Returns
        -------
        dict
            Dictionary of the form described above."""
        if (method.lower(), scaled) != ("chisquare", True):
            raise NotImplementedError("Not implemented")
        lmparamdict = self.fitter.pars["Fit"][self.name]
        return_dict = {
            param_name: [
                lmparamdict[param_name].value,
                lmparamdict[param_name].unc,
            ]
            for param_name in lmparamdict.keys()
        }
        return return_dict

    def get_result_frame(
        self,
        method: str = "chisquare",
        selected: bool = False,
        bounds: bool = False,
        vary: bool = False,
        scaled: bool = True,
    ) -> pd.DataFrame:
        """Returns the data from the fit in a pandas DataFrame.

        Parameters
        ----------
        method: str, optional
            Selects which fitresults have to be loaded. Can be 'chisquare' or
            'mle'. Defaults to 'chisquare', and only 'chisquare'.
        selected: list of strings, optional
            Selects the parameters that have any string in the list
            as a substring in their name. Set to *None* to select
            all parameters. Defaults to None, and only None.
        bounds: boolean, optional
            Selects if the boundary also has to be given. Defaults to
            False, and onlyb False.
        vary: boolean, optional
            Selects if only the parameters that have been varied have to
            be supplied. Defaults to False, and only False.
        scaled: boolean, optional
            Sets the uncertainty scaling with the reduced chisquare value. Default to True, and only True

        Returns
        -------
        resultframe: DataFrame
            Dateframe with MultiIndex, using the variable names as main column names
            and the two rows under for the value and the uncertainty"""
        result_dict = self.get_result_dict(method=method, scaled=scaled)
        return_frame = pd.DataFrame.from_dict(result_dict)
        return return_frame


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

    def __init__(
        self,
        models: list,
        background_params: list,
        name: str = "sum",
        source_name: str = "source",
    ):
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

    def set_variation(self, varyDict: dict):
        raise NotImplementedError("Do this at the HFSModel level")

    def f(self, x: ArrayLike) -> ArrayLike:
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

    def __call__(self, x):
        return self.f(x)

    def chisquare_fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        yerr: Union[ArrayLike, callable] = None,
        xerr: ArrayLike = None,
        func: callable = None,
        verbose: bool = False,
        hessian: bool = False,
        method: str = "leastsq",
    ) -> Tuple[bool, str]:
        """Perform a fit of this model to the data provided in this function.

        Parameters
        ----------
        x : ArrayLike
            x-values of the data points
        y : ArrayLike
            y-values of the data-points
        yerr : Union[ArrayLike, callable], optional
            1-sigma error on the y-values, values or function to be used on f(x),
            by default None=sqrt(f(x))
        xerr : ArrayLike, optional
            1-sigma error on the x-values, by default None
        func : callable, optional
            Not implemented
        verbose : bool, optional
            Not implemented, by default False
        hessian : bool, optional
            Not implemented, by default False
        method : str, optional
            Selects the method used by the :func:`lmfit.minimizer`, by default 'leastsq'

        Returns
        -------
        Tuple[bool, str]
            Returns a boolean indicating the success of the fit, and the message accompanying it.

        Raises
        ------
        NotImplementedError
            When the chosen options for func, verbose and Hessian result is not implemented.
        """
        if (func, verbose, hessian) != (None, False, False):
            raise NotImplementedError("Not implemented")
        datasource = Source(x, y, yerr=yerr, name="Fit")
        for model in self.models:
            datasource.addModel(model)
        step_bkg = PiecewiseConstant(
            self.background_params["values"],
            self.background_params["bounds"],
            name="bkg",
        )
        self.models.append(step_bkg)
        datasource.addModel(step_bkg)
        self.fitter = Fitter()
        self.fitter.addSource(datasource)
        self.fitter.fit(method=method)
        return self.fitter.result.success, self.fitter.result.message

    def display_chisquare_fit(self, scaled: bool = True, **kwargs) -> str:
        """Generate a report of the fitting results.

        The report contains the best-fit values for the parameters and their uncertainties and correlations.

        Parameters
        ----------
        scaled: bool, optional
            Whether the errors are scaled with reduced chisquared, by default True, and only True
        show_correl : bool, optional
            Whether to show a list of sorted correlations, by default False
        min_correl : float, optional
            Smallest correlation in absolute value to show, by default 0.1

        Returns
        -------
        str
            Multi-line text of fit report.
        """
        if not scaled:
            raise NotImplementedError("Not implemented")
        return lm.fit_report(self.fitter.result, **kwargs)

    def get_result(
        self, selection: str = "chisquare"
    ) -> Tuple[list, list, list]:
        """Return the variable names, values and estimated error bars for the
        parameters as seperate lists.

        Parameters
        ----------
        selection: string, optional
            Selects if the chisquare ('chisquare' or 'any') or MLE values are
            used. Defaults to 'chisquare', and chisquare only

        Returns
        -------
        names, values, uncertainties: tuple of lists
            Returns a 3-tuple of lists containing the names of the parameters. The first list each tuple element contains the names/values/uncertainties of the first model added to the summodel, etc.
            The last list in each tuple element contains the names/values/uncertainties for the step background
            The values and the estimated uncertainties are always scaled with the reduced chisquared.
        """
        varnames = []
        varvalues = []
        varunc = []
        for model in self.models:
            lmparamdict = self.fitter.pars["Fit"][model.name]
            varnames.append(list(lmparamdict.keys()))
            varvalues.append(
                [
                    lmparamdict[param_name].value
                    for param_name in lmparamdict.keys()
                ]
            )
            varunc.append(
                [
                    lmparamdict[param_name].unc
                    for param_name in lmparamdict.keys()
                ]
            )
        return varnames, varvalues, varunc

    def get_result_dict(
        self, method: str = "chisquare", scaled: bool = True
    ) -> dict:
        """Returns the fitted parameters in a dictionary of the form {name of model in summodel : {name: [value, uncertainty]}}. Background values are under key 'bkg' in dictionary.

        Parameters
        ----------
        method: {'chisquare', 'mle'}
            Selects which parameters have to be returned, by default 'chisquare', and only 'chisquare'
        scaled: boolean
            Selects if, in case of chisquare parameters, the uncertainty
            has to be scaled by sqrt(reduced_chisquare). Defaults to True, and only True

        Returns
        -------
        dict
            Dictionary of the form described above."""
        if (method.lower(), scaled) != ("chisquare", True):
            raise NotImplementedError("Not implemented")
        return_dict = dict()
        for model in self.models:
            lmparamdict = self.fitter.pars["Fit"][model.name]
            return_dict[model.name] = {
                param_name: [
                    lmparamdict[param_name].value,
                    lmparamdict[param_name].unc,
                ]
                for param_name in lmparamdict.keys()
            }
        return return_dict

    def get_result_frame(
        self,
        method: str = "chisquare",
        selected: bool = False,
        bounds: bool = False,
        vary: bool = False,
        scaled: bool = True,
    ) -> pd.DataFrame:
        """Returns the data from the fit in a pandas DataFrame.

        Parameters
        ----------
        method: str, optional
            Selects which fitresults have to be loaded. Can be 'chisquare' or
            'mle'. Defaults to 'chisquare', and only 'chisquare'.
        selected: list of strings, optional
            Selects the parameters that have any string in the list
            as a substring in their name. Set to *None* to select
            all parameters. Defaults to None, and only None.
        bounds: boolean, optional
            Selects if the boundary also has to be given. Defaults to
            False, and onlyb False.
        vary: boolean, optional
            Selects if only the parameters that have been varied have to
            be supplied. Defaults to False, and only False.
        scaled: boolean, optional
            Sets the uncertainty scaling with the reduced chisquare value. Default to True, and only True

        Returns
        -------
        resultframe: DataFrame
            Dateframe with MultiIndex, using the model name + variable names as main column names
            and the two rows under for the value and the uncertainty"""
        result_dict = self.get_result_dict(method=method, scaled=scaled)
        return_frame = pd.DataFrame.from_dict(result_dict[self.models[0].name])
        return_frame = return_frame.add_prefix(f"{self.models[0].name}_")
        for model in self.models[1:]:
            df_to_add = pd.DataFrame.from_dict(result_dict[model.name])
            df_to_add = df_to_add.add_prefix(f"{model.name}_")
            return_frame = pd.concat([return_frame, df_to_add], axis=1)
        return return_frame


def chisquare_fit(
    model: Union["HFSModel", "SumModel"],
    x: ArrayLike,
    y: ArrayLike,
    yerr: Union[ArrayLike, callable],
    xerr: ArrayLike = None,
    method: str = "leastsq",
):
    """Perform a fit of the provided model to the data provided in this function.

    Parameters
    ----------
    model :
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
        Show correlations between fitted parameters in fit message, by default True

    Returns
    -------
    Instance of Fitter"""
    return model.chisquare_fit(x=x, y=y, yerr=yerr, xerr=xerr, method=method)
