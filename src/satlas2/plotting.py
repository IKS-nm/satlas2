"""
Functions for the generation of plots related to the fitting results.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""
import copy
from typing import List, Optional, Tuple

from numpy.typing import ArrayLike

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import uncertainties as u
from scipy import optimize
from scipy.stats import chi2

from .core import Fitter, _split_name
from .overwrite import SATLASHDFBackend

inv_color_list = [
    "#7acfff",
    "#fff466",
    "#00c48f",
    "#ff8626",
    "#ff9cd3",
    "#0093e6",
]
color_list = [c for c in reversed(inv_color_list)]
cmap = mpl.colors.ListedColormap(color_list)
cmap.set_over(color_list[-1])
cmap.set_under(color_list[0])
invcmap = mpl.colors.ListedColormap(inv_color_list)
invcmap.set_over(inv_color_list[-1])
invcmap.set_under(inv_color_list[0])

__all__ = [
    "generateChisquareMap",
    "generateCorrelationPlot",
    "generateWalkPlot",
]

# Percentiles of the median and the 1 sigma interval
_QUANTILES = [15.87, 50, 84.13]
# Probability mass inside the 1, 2 and 3 sigma contours of a 2D Gaussian
_MASSES_2D = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2)
# Number of histogram bins when Scott's rule cannot be applied
_FALLBACK_BINS = 50
_MARKER_COLOR = "#0093e6"


def _parameterLabels(
    full_names: List[str], source: bool, model: bool, separator: str
) -> List[str]:
    """Labels of the parameters, optionally with the source and model name."""
    labels = []
    for full_name in full_names:
        source_name, model_name, parameter = _split_name(full_name)
        parts = [source_name] * source + [model_name] * model + [parameter]
        labels.append(separator.join(parts))
    return labels


def _readWalk(filename: str, burnin: int, thin: int, autoprocess: bool, flat: bool):
    """Parameter names, chain, burn-in and thinning of a saved random walk.
    With autoprocess, the burn-in is twice the longest and the thinning half
    the shortest autocorrelation time."""
    reader = SATLASHDFBackend(filename)
    if autoprocess:
        tau = reader.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        thin = max(1, int(0.5 * np.min(tau)))
    chain = reader.get_chain(flat=flat, discard=burnin, thin=thin)
    return list(reader.labels), chain, burnin, thin


def _selectParameters(
    filter: Optional[List[str]], labels: List[str], full_names: List[str]
) -> List[Tuple[str, str]]:
    """(label, full name) of the parameters whose label contains one of the
    filter strings, in the order of the filter; all parameters without filter."""
    pairs = list(zip(labels, full_names))
    if filter is None:
        return pairs
    return [(label, full) for f in filter for (label, full) in pairs if f in label]


def _scottBins(x: np.ndarray, reduction: int = 1) -> int:
    """Number of histogram bins following Scott's rule for the bin width, at
    most 1000, divided by reduction."""
    with np.errstate(divide="ignore", invalid="ignore"):
        number = np.ptp(x) / (3.5 * np.std(x) / x.size ** (1 / 3))
    if not np.isfinite(number):
        return _FALLBACK_BINS
    return max(1, int(min(int(number), 1000) / reduction))


def _credibleLevels(H: np.ndarray, masses: ArrayLike = _MASSES_2D) -> np.ndarray:
    """Values of a 2D histogram at which the bins above that value hold the
    given fractions of the total, for drawing credible regions."""
    flat = np.sort(H.ravel())[::-1]
    cumulative = np.cumsum(flat)
    cumulative /= cumulative[-1]
    levels = []
    for mass in masses:
        inside = flat[cumulative <= mass]
        levels.append(inside[-1] if inside.size else flat[0])
    return np.array(levels)


def _intervalTitle(name: str, value: float, minus: float, plus: float) -> str:
    """Title with the value and its asymmetric uncertainty, rounded to two
    significant digits of the uncertainty."""
    up = "{:.2ug}".format(u.ufloat(value, plus))
    down = "{:.2ug}".format(u.ufloat(value, minus))
    rounded = up.split("+/-")[0].split("(")[-1]
    plus_text = up.split("+/-")[1].split(")")[0]
    minus_text = down.split("+/-")[1].split(")")[0]
    if "e" in up or "e" in down:
        exponent = up.split("e")[-1]
        return "{}\n$({}_{{-{}}}^{{+{}}})e{}$".format(
            name, rounded, minus_text, plus_text, exponent
        )
    return "{}\n${}_{{-{}}}^{{+{}}}$".format(name, rounded, minus_text, plus_text)


def _make_axes_grid(
    no_variables,
    width=6,
    height=6,
    cbar=True,
    left=0.1,
    right=0.9,
    top=0.9,
    bottom=0.1,
):
    """Makes a triangular grid of axes, with a colorbar axis next to it.

    Parameters
    ----------
    no_variables: int
        Number of variables for which to generate a figure.
    padding: float
        Padding around the figure (in cm).
    cbar_size: float
        Width of the colorbar (in cm).
    axis_padding: float
        Padding between axes (in cm).

    Returns
    -------
    fig, axes, cbar: tuple
        Tuple containing the figure, a 2D-array of axes and the colorbar axis.
    """

    fig = plt.figure(constrained_layout=True, figsize=(width, height))
    width_ratios = [1] * no_variables
    if cbar:
        width_ratios.extend([0.1] * 2)
    gs = gridspec.GridSpec(
        nrows=no_variables,
        ncols=no_variables + 2 * cbar,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=0,
        hspace=0,
        figure=fig,
        width_ratios=width_ratios,
    )

    # Pre-allocate a 2D-array to hold the axes.
    axes = np.array(
        [[None for _ in range(no_variables)] for _ in range(no_variables)],
        dtype="object",
    )

    for i, I in zip(range(no_variables), reversed(range(no_variables))):
        for j in reversed(range(no_variables)):
            # Only create axes on the lower triangle.
            if I + j < no_variables:
                # Share the x-axis with the plot on the diagonal,
                # directly above the plot.
                sharex = axes[j, j] if i != j else None
                # Share the y-axis among the 2D maps along one row,
                # but not the plot on the diagonal!
                sharey = axes[i, i - 1] if (i != j and i - 1 != j) else None
                a = fig.add_subplot(gs[i, j], sharex=sharex, sharey=sharey)
                a.label_outer()
                plt.setp(a.xaxis.get_majorticklabels(), rotation=45)
                plt.setp(a.yaxis.get_majorticklabels(), rotation=45)
            else:
                a = None
            if i == j:
                a.set_yticks([])
                a.set_yticklabels([])
            axes[i, j] = a

    axes = np.array(axes)
    if cbar:
        cbar = fig.add_subplot(gs[:, -1])
    else:
        cbar = None
    return fig, axes, cbar


def generateChisquareMap(
    fitter: Fitter,
    filter: Optional[List[str]] = None,
    method: str = "chisquare",
    resolution_diag: int = 15,
    resolution_map: int = 15,
    fit_kws: dict = {},
    source: bool = False,
    model: bool = True,
):
    """:meta private:
    Generates a correlation map for either the chisquare or the MLE method.
    On the diagonal, the chisquare or loglikelihood is drawn as a function of one fixed parameter.
    Refitting to the data each time gives the points on the line. A dashed line is drawn on these
    plots, with the intersection with the plots giving the correct confidence interval for the
    parameter. In solid lines, the interval estimated by the fitting routine is drawn.
    On the offdiagonal, two parameters are fixed and the model is again fitted to the data.
    The change in chisquare/loglikelihood is mapped to 1, 2 and 3 sigma contourmaps.

    Parameters
    ----------
    fitter: :class:`.Fitter`
        Fitter instance for which the chisquare map must be created.

    Other parameters
    ----------------
    filter: list of strings
        Only the parameters matching the names given in this list will be used
        to generate the maps.
    resolution_diag: int
        Number of points for the line plot on each diagonal.
    resolution_map: int
        Number of points along each dimension for the meshgrids.
    fit_kws: dictionary
        Dictionary of keywords to pass on to the fitting routine.
    npar: int
        Number of parameters for which simultaneous predictions need to be made.
        Influences the uncertainty estimates from the parabola."""

    title = "{}\n${}_{{-{}}}^{{+{}}}$"
    title_e = "{}\n$({}_{{-{}}}^{{+{}}})e{}$"

    try:
        orig_value = fitter.chisqr
    except AttributeError:
        fitter.fit(**fit_kws)
        orig_value = fitter.chisqr
    if method.lower().startswith("llh"):
        orig_value = fitter.llh_result
    result = copy.deepcopy(fitter.result)
    orig_params = copy.deepcopy(fitter.lmpars)

    ranges = {}

    param_names = []
    no_params = 0
    for p in orig_params:
        if orig_params[p].vary and (
            filter is None or any([f in p for f in filter])
        ):
            no_params += 1
            param_names.append(p)
    fig, axes, cbar = _make_axes_grid(
        no_params, axis_padding=0, cbar=no_params > 1
    )

    split_names = [name.split("___") for name in param_names]
    sources = [name[0] for name in split_names]
    models = [name[1] for name in split_names]
    var_names = [name[2] for name in split_names]
    to_be_combined = [var_names]
    if model:
        to_be_combined.insert(0, models)
    if source:
        to_be_combined.insert(0, sources)

    var_names = [" ".join(tbc) for tbc in zip(*to_be_combined)]

    # Make the plots on the diagonal: plot the chisquare/likelihood
    # for the best fitting values while setting one parameter to
    # a fixed value.
    saved_params = copy.deepcopy(fitter.lmpars)
    for i in range(no_params):
        params = copy.deepcopy(saved_params)
        ranges[param_names[i]] = {}

        # Set the y-ticklabels.
        ax = axes[i, i]
        ax.set_title(param_names[i])
        if i == no_params - 1:
            if method.lower().startswith("chisquare"):
                ax.set_ylabel(r"$\Delta\chi^2$")
            else:
                ax.set_ylabel(r"$\Delta\mathcal{L}$")
                fit_kws["llh_selected"] = True

        # Select starting point to determine error widths.
        value = orig_params[param_names[i]].value
        stderr = orig_params[param_names[i]].stderr
        stderr = stderr if stderr is not None else 0.01 * np.abs(value)
        stderr = stderr if stderr != 0 else 0.01 * np.abs(value)

        right = value + stderr
        left = value - stderr
        params[param_names[i]].vary = False

        ranges[param_names[i]]["left_val"] = 3 * left - 2 * value
        ranges[param_names[i]]["right_val"] = 3 * right - 2 * value
        value_range = np.linspace(
            3 * left - 2 * value, right * 3 - 2 * value, resolution_diag
        )
        chisquare = np.zeros(len(value_range))
        # Calculate the new value, and store it in the array. Update the progressbar.
        # with tqdm.tqdm(value_range, desc=param_names[i], leave=True) as pbar:
        for j, v in enumerate(value_range):
            params[param_names[i]].value = v
            fitter.lmpars = params
            fitter.fit(prepFit=False, **fit_kws)
            if fitter.llh_result is not None:
                chisquare[j] = fitter.llh_result - orig_value
            else:
                chisquare[j] = fitter.chisqr - orig_value
                # pbar.update(1)
        # Plot the result
        ax.plot(value_range, chisquare, color="k")

        c = "#0093e6"
        ax.axvline(right, ls="dashed", color=c)
        ax.axvline(left, ls="dashed", color=c)
        ax.axvline(value, ls="dashed", color=c)
        up = "{:.2ug}".format(u.ufloat(value, stderr))
        down = "{:.2ug}".format(u.ufloat(value, stderr))
        val = up.split("+/-")[0].split("(")[-1]
        r = up.split("+/-")[1].split(")")[0]
        l = down.split("+/-")[1].split(")")[0]
        if "e" in up or "e" in down:
            ex = up.split("e")[-1]
            ax.set_title(title_e.format(var_names[i], val, l, r, ex))
        else:
            ax.set_title(title.format(var_names[i], val, l, r))
        # Restore the parameters.
        fitter.lmpars = orig_params

    for i, j in zip(*np.tril_indices_from(axes, -1)):
        params = copy.deepcopy(orig_params)
        ax = axes[i, j]
        x_name = param_names[j]
        y_name = param_names[i]
        if j == 0:
            ax.set_ylabel(var_names[i])
        if i == no_params - 1:
            ax.set_xlabel(var_names[j])
        right = ranges[x_name]["right_val"]
        left = ranges[x_name]["left_val"]
        x_range = np.linspace(left, right, resolution_map)

        right = ranges[y_name]["right_val"]
        left = ranges[y_name]["left_val"]
        y_range = np.linspace(left, right, resolution_map)

        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros(X.shape)
        i_indices, j_indices = np.indices(Z.shape)
        params[param_names[i]].vary = False
        params[param_names[j]].vary = False

        for k, l in zip(i_indices.flatten(), j_indices.flatten()):
            x = X[k, l]
            y = Y[k, l]
            params[param_names[j]].value = x
            params[param_names[i]].value = y
            fitter.lmpars = params
            fitter.fit(prepFit=False, **fit_kws)
            if fitter.llh_result is not None:
                Z[k, l] = (fitter.llh_result - orig_value) * 2
            else:
                Z[k, l] = fitter.chisqr - orig_value

        Z = -Z
        bounds = []
        for bound in [0.997300204, 0.954499736, 0.682689492]:
            chifunc = (
                lambda x: chi2.cdf(x, 1) - bound
            )  # Calculate 1 sigma boundary
            bounds.append(-optimize.root(chifunc, 1).x[0])
        bounds.append(0)
        bounds = np.array(bounds)
        norm = mpl.colors.BoundaryNorm(bounds, invcmap.N)
        contourset = ax.contourf(X, Y, Z, bounds, cmap=invcmap, norm=norm)
        fitter.lmpars = copy.deepcopy(orig_params)
    try:
        cbar = plt.colorbar(contourset, cax=cbar, orientation="vertical")
        cbar.ax.yaxis.set_ticks([-7.5, -4.5, -1.5])
        cbar.ax.set_yticklabels([r"3$\sigma$", r"2$\sigma$", r"1$\sigma$"])
    except:
        pass
    for a in axes.flatten():
        if a is not None:
            for label in a.get_xticklabels()[::2]:
                label.set_visible(False)
            for label in a.get_yticklabels()[::2]:
                label.set_visible(False)
    fitter.result = result
    fitter.updateInfo()
    return fig, axes, cbar


def generateCorrelationPlot(
    filename: str,
    filter: Optional[List[str]] = None,
    bins: Optional[int] = None,
    burnin: int = 0,
    thin: int = 1,
    autoprocess: bool = False,
    source: bool = True,
    model: bool = True,
    binreduction: int = 1,
    bin2dreduction: int = 1,
    progress: bool = False,
    width: float = 6,
    height: float = 6,
    left: float = 0.15,
    right: float = 0.95,
    top: float = 0.85,
    bottom: float = 0.15,
) -> Tuple[plt.Figure, Tuple[plt.Axes], plt.Axes]:
    r"""Given the random walk data, creates a triangle plot: distribution of
    a single parameter on the diagonal axes, 2D contour plots with 1, 2 and
    3 sigma contours on the off-diagonal. The 1-sigma limits based on the
    percentile method are also indicated, as well as added to the title.

    Parameters
    ----------
    filename : str
        Filename for the h5 file containing the data from the walk.
    filter : List[str], optional
        Only this list of columns is used for the plot, by default None.
    bins : int, optional
        Use this number of bins for the plotting.
        Applies the same number of bins for each parameter.
        If supplied as a list, length must match the number of
        parameters. By default None.
    burnin : int, optional
        Number of initial steps from the random walk to be discarded,
        by default 0.
    thin : int, optional
        Take only every ``thin`` steps from the chain. (default: ``1``)
    autoprocess : bool, optional
        Based on the autocorrelation time of the random walk, perform an
        automatic burn-in and thinning estimate, by default False.
    source : bool, optional
        Add the source name to the plot titles, by default True.
    model : bool, optional
        Add the model name to the plot titles, by default True.
    binreduction : int, optional
        Reduces the amount of bins in the 1D case by this factor,
        by default 1.
    bin2dreduction : int, optional
        Further reduces the amount of bins in the 2D case by this factor,
        by default 1.
    progress : bool, optional
        Show a progress bar of processing the parameters, by default False.
    width : float, optional
        Width in inches of the figure, by default 6
    height : float, optional
        Height in inches of the figure, by default 6
    left : float, optional
        Extent of the left of the figure, in fraction, by default 0.15
    right : float, optional
        Extent of the right of the figure, in fraction, by default 0.95
    top : float, optional
        Extent of the top of the figure, in fraction, by default 0.85
    bottom : float, optional
        Extent of the bottom of the figure, in fraction, by default 0.15

    Returns
    -------
    Tuple[plt.Figure, Tuple[plt.Axes], plt.Axes]
        Tuple containing the figure, the individual axes, and the colorbar axis.

    Note
    ----
    When estimated automatically, the ``burnin`` and ``thin`` are set to
    respectively

    .. math::
        2\cdot\textrm{max}\left(\tau\right)

    and

    .. math::
        \textrm{min}\left(\tau\right)/2"""
    full_names, data, burnin, thin = _readWalk(
        filename, burnin, thin, autoprocess, flat=True
    )
    labels = _parameterLabels(full_names, source, model, "\n")
    selected = _selectParameters(filter, labels, full_names)
    columns = {full: full_names.index(full) for _, full in selected}
    if not isinstance(bins, list):
        bins = [bins for _ in selected]

    with tqdm.tqdm(
        total=len(selected) + (len(selected) ** 2 - len(selected)) / 2,
        leave=True,
        disable=not progress,
    ) as pbar:
        fig, axes, cbar = _make_axes_grid(
            len(selected),
            width=width,
            height=height,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )
        fig.set_layout_engine(None)

        # distribution of each parameter on the diagonal
        ranges = {}
        for i, (name, full_name) in enumerate(selected):
            pbar.set_description(name)
            ax = axes[i, i]
            x = data[:, columns[full_name]]
            if bins[i] is None:
                bins[i] = _scottBins(x, binreduction)
            ax.hist(x, int(bins[i]), histtype="step", color="k")
            ranges[full_name] = (np.min(x), np.max(x), bins[i])

            q16, q50, q84 = np.percentile(x, _QUANTILES)
            ax.set_title(
                _intervalTitle(name, q50, np.abs(q50 - q16), np.abs(q84 - q50))
            )
            for q in (q16, q50, q84):
                ax.axvline(q, ls="dashed", color=_MARKER_COLOR)
            pbar.update(1)

        # credible regions of each pair of parameters below the diagonal
        contourset = None
        for i, j in zip(*np.tril_indices_from(axes, -1)):
            x_name, x_fullname = selected[j]
            y_name, y_fullname = selected[i]
            pbar.set_description(", ".join([x_name, y_name]))
            ax = axes[i, j]
            if j == 0:
                ax.set_ylabel(y_name)
            if i == len(selected) - 1:
                ax.set_xlabel(x_name)
            x_min, x_max, x_bins = ranges[x_fullname]
            y_min, y_max, y_bins = ranges[y_fullname]
            H, X, Y = np.histogram2d(
                data[:, columns[x_fullname]],
                data[:, columns[y_fullname]],
                bins=(
                    np.linspace(x_min, x_max, int(x_bins / bin2dreduction) + 1),
                    np.linspace(y_min, y_max, int(y_bins / bin2dreduction) + 1),
                ),
            )
            X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
            H = (H - np.min(H)) / (np.max(H) - np.min(H))

            bounds = np.unique(np.concatenate([[H.max()], _credibleLevels(H)])[::-1])
            norm = mpl.colors.BoundaryNorm(bounds, invcmap.N)
            contourset = ax.contourf(X1, Y1, H.T, bounds, cmap=invcmap, norm=norm)
            pbar.update(1)
        cbar = _sigmaColorbar(contourset, cbar)
    return fig, axes, cbar


def _sigmaColorbar(contourset, cax):
    """Colorbar labelled with the 1, 2 and 3 sigma regions. Returns None if
    no regions were drawn, or if the last pair of parameters has fewer than
    three distinct regions (the colorbar is then drawn without labels)."""
    if contourset is None or cax is None:
        return None
    try:
        cbar = plt.colorbar(contourset, cax=cax, orientation="vertical")
        ticks = cbar.ax.get_yticks()
        ticks = ticks[:-1] + (ticks[1:] - ticks[:-1]) / 2
        cbar.ax.yaxis.set_ticks(ticks)
        cbar.ax.set_yticklabels([r"3$\sigma$", r"2$\sigma$", r"1$\sigma$"])
    except ValueError:
        return None
    return cbar


def generateWalkPlot(
    filename: str,
    filter: Optional[List[str]] = None,
    burnin: int = 0,
    thin: int = 1,
    autoprocess: bool = False,
    source: bool = False,
    model: bool = True,
    progress: bool = False,
) -> Tuple[plt.Figure, Tuple[plt.Axes]]:
    r"""Given the random walk data, the random walk for the selected parameters
    is plotted.

    Parameters
    ----------
    filename : str
        Filename for the h5 file containing the data from the walk.
    filter : List[str], optional
        Only this list of columns is used for the plot, by default None.
    burnin : int, optional
        Number of initial steps from the random walk to be discarded,
        by default 0.
    thin : int, optional
        Take only every ``thin`` steps from the chain. (default: ``1``)
    autoprocess : bool, optional
        Based on the autocorrelation time of the random walk, perform an
        automatic burn-in and thinning estimate, by default False.
    source : bool, optional
        Add the source name to the plot titles, by default False.
    model : bool, optional
        Add the model name to the plot titles, by default True.
    progress : bool, optional
        Show a progress bar of processing the parameters, by default False.

    Returns
    -------
    Tuple[plt.Figure, Tuple[plt.Axes]]
        Tuple containing the figure and the individual axes.

    Note
    ----
    When estimated automatically, the ``burnin`` and ``thin`` are set to
    respectively

    .. math::
        2\cdot\textrm{max}\left(\tau\right)

    and

    .. math::
        \textrm{min}\left(\tau\right)/2"""
    full_names, data, burnin, thin = _readWalk(
        filename, burnin, thin, autoprocess, flat=False
    )
    labels = _parameterLabels(full_names, source, model, "\n")
    selected = _selectParameters(filter, labels, full_names)
    steps = np.arange(data.shape[0]) * thin + burnin

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(nrows=len(selected), ncols=1, figure=fig)
    axes = []
    for i, (name, full_name) in enumerate(
        tqdm.tqdm(selected, leave=True, disable=not progress)
    ):
        ax = fig.add_subplot(gs[i, 0], sharex=axes[-1] if axes else None)
        ax.label_outer()
        x = data[:, :, full_names.index(full_name)]
        ax.plot(steps, x, alpha=0.3, color="gray")
        ax.set_ylabel(name)
        ax.axhline(np.percentile(x, [50.0]), color="k")
        axes.append(ax)
    axes[-1].set_xlabel("Step")
    return fig, axes
