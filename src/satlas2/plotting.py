"""
Functions for the generation of plots related to the fitting results.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""
import copy
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import uncertainties as u
from scipy import optimize
from scipy.stats import chi2

from .core import Fitter
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
    """Given the random walk data, creates a triangle plot: distribution of
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
    Left : float, optional
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
        2\cdot\\textrm{max}\\left(\\tau\\right)

    and

    .. math::
        \\textrm{min}\\left(\\tau\\right)/2"""

    reader = SATLASHDFBackend(filename)
    var_names = list(reader.labels)
    split_names = [name.split("___") for name in var_names]
    sources = [name[0] for name in split_names]
    models = [name[1] for name in split_names]
    var_names = [name[2] for name in split_names]
    to_be_combined = [var_names]
    if model:
        to_be_combined.insert(0, models)
    if source:
        to_be_combined.insert(0, sources)

    var_names = ["\n".join(tbc) for tbc in zip(*to_be_combined)]
    full_names = list(reader.labels)

    if autoprocess:
        tau = reader.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
    data = reader.get_chain(flat=True, discard=burnin, thin=thin)

    if filter is not None:
        filter = [
            (c, n)
            for f in filter
            for (c, n) in zip(var_names, full_names)
            if f in c
        ]
    else:
        filter = list(zip(var_names, full_names))
    with tqdm.tqdm(
        total=len(filter) + (len(filter) ** 2 - len(filter)) / 2,
        leave=True,
        disable=not progress,
    ) as pbar:
        fig, axes, cbar = _make_axes_grid(
            len(filter),
            width=width,
            height=height,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )
        fig.set_layout_engine(None)

        metadata = {}
        if not isinstance(bins, list):
            bins = [bins for _ in filter]
        for i, val in enumerate(filter):
            name, full_name = val
            pbar.set_description(name)
            ax = axes[i, i]
            bin_index = i
            i = full_names.index(full_name)
            x = data[:, i]

            if bins[bin_index] is None:
                width = (
                    3.5 * np.std(x) / x.size ** (1 / 3)
                )  # Scott's rule for binwidth
                bins[bin_index] = int(
                    min(int(x.ptp() / width), 1000) / binreduction
                )
            try:
                (
                    n,
                    b,
                    p,
                ) = ax.hist(
                    x, int(bins[bin_index]), histtype="step", color="k"
                )
            except TypeError:
                bins[bin_index] = 50
                (
                    n,
                    b,
                    p,
                ) = ax.hist(
                    x, int(bins[bin_index]), histtype="step", color="k"
                )
            q = [15.87, 50, 84.13]
            q16, q50, q84 = np.percentile(x, q)
            metadata[full_name] = {
                "bins": bins[bin_index],
                "min": x.min(),
                "max": x.max(),
            }

            title = "{}\n${}_{{-{}}}^{{+{}}}$"
            title_e = "{}\n$({}_{{-{}}}^{{+{}}})e{}$"
            up = "{:.2ug}".format(u.ufloat(q50, np.abs(q84 - q50)))
            down = "{:.2ug}".format(u.ufloat(q50, np.abs(q50 - q16)))
            param_val = up.split("+/-")[0].split("(")[-1]
            r = up.split("+/-")[1].split(")")[0]
            l = down.split("+/-")[1].split(")")[0]
            if "e" in up or "e" in down:
                ex = up.split("e")[-1]
                t = ax.set_title(title_e.format(name, param_val, l, r, ex))
            else:
                t = ax.set_title(title.format(name, param_val, l, r))

            qvalues = [q16, q50, q84]
            c = "#0093e6"
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=c)
            pbar.update(1)

        for i, j in zip(*np.tril_indices_from(axes, -1)):
            x_name, x_fullname = filter[j]
            y_name, y_fullname = filter[i]
            pbar.set_description(", ".join([x_name, y_name]))
            ax = axes[i, j]
            if j == 0:
                ax.set_ylabel(y_name)
            if i == len(filter) - 1:
                ax.set_xlabel(x_name)
            j = full_names.index(x_fullname)
            i = full_names.index(y_fullname)
            x = data[:, j]
            y = data[:, i]
            x_min, x_max, x_bins = (
                metadata[x_fullname]["min"],
                metadata[x_fullname]["max"],
                metadata[x_fullname]["bins"],
            )
            y_min, y_max, y_bins = (
                metadata[y_fullname]["min"],
                metadata[y_fullname]["max"],
                metadata[y_fullname]["bins"],
            )
            X = np.linspace(x_min, x_max, int(x_bins / bin2dreduction) + 1)
            Y = np.linspace(y_min, y_max, int(y_bins / bin2dreduction) + 1)
            H, X, Y = np.histogram2d(
                x.flatten(), y.flatten(), bins=(X, Y), weights=None
            )
            X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
            X, Y = X[:-1], Y[:-1]
            H = (H - H.min()) / (H.max() - H.min())

            Hflat = H.flatten()
            inds = np.argsort(Hflat)[::-1]
            Hflat = Hflat[inds]
            sm = np.cumsum(Hflat)
            sm /= sm[-1]
            levels = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2)
            V = np.empty(len(levels))
            for i, v0 in enumerate(levels):
                try:
                    V[i] = Hflat[sm <= v0][-1]
                except:
                    V[i] = Hflat[0]

            bounds = np.unique(np.concatenate([[H.max()], V])[::-1])
            norm = mpl.colors.BoundaryNorm(bounds, invcmap.N)

            contourset = ax.contourf(
                X1, Y1, H.T, bounds, cmap=invcmap, norm=norm
            )
            pbar.update(1)
        try:
            cbar = plt.colorbar(contourset, cax=cbar, orientation="vertical")
            ticks = cbar.ax.get_yticks()
            dfticks = (ticks[1:] - ticks[:-1]) / 2
            ticks = ticks[:-1] + dfticks
            cbar.ax.yaxis.set_ticks(ticks)
            cbar.ax.set_yticklabels([r"3$\sigma$", r"2$\sigma$", r"1$\sigma$"])
        except:
            cbar = None
    return fig, axes, cbar


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
    """Given the random walk data, the random walk for the selected parameters
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
        2\cdot\\textrm{max}\\left(\\tau\\right)

    and

    .. math::
        \\textrm{min}\\left(\\tau\\right)/2"""
    reader = SATLASHDFBackend(filename)
    var_names = reader.labels
    split_names = [name.split("___") for name in var_names]
    sources = [name[0] for name in split_names]
    models = [name[1] for name in split_names]
    var_names = [name[2] for name in split_names]
    to_be_combined = [var_names]
    if model:
        to_be_combined.insert(0, models)
    if source:
        to_be_combined.insert(0, sources)

    var_names = ["\n".join(tbc) for tbc in zip(*to_be_combined)]
    full_names = list(reader.labels)

    if autoprocess:
        tau = reader.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
    data = reader.get_chain(flat=False, discard=burnin, thin=thin)
    plot_x = np.arange(data.shape[0]) * thin + burnin

    if filter is not None:
        filter = [
            (c, n)
            for f in filter
            for (c, n) in zip(var_names, full_names)
            if f in c
        ]
    else:
        filter = list(zip(var_names, full_names))
    with tqdm.tqdm(
        total=len(filter), leave=True, disable=not progress
    ) as pbar:
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(nrows=len(filter), ncols=1, figure=fig)

        old_ax = None
        axes = []
        for i, val in enumerate(filter):
            ax = fig.add_subplot(gs[i, 0], sharex=old_ax)
            old_ax = ax
            ax.label_outer()
            name, full_name = val
            pbar.set_description(name)
            i = full_names.index(full_name)
            x = data[:, :, i]
            q50 = np.percentile(x, [50.0])
            ax.plot(plot_x, x, alpha=0.3, color="gray")
            ax.set_ylabel(name)
            ax.axhline(q50, color="k")
            axes.append(ax)
            pbar.update(1)
        ax.set_xlabel("Step")
    pbar.close()
    return fig, axes
