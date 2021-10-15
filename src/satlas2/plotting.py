"""
Functions for the generation of plots related to the fitting results.

.. moduleauthor:: Wouter Gins <wouter.a.gins@jyu.fi>
"""
import copy

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import uncertainties as u
from scipy import optimize
from scipy.stats import chi2
import tqdm
from .overwrite import SATLASHDFBackend

inv_color_list = ['#7acfff', '#fff466', '#00c48f', '#ff8626', '#ff9cd3', '#0093e6']
color_list = [c for c in reversed(inv_color_list)]
cmap = mpl.colors.ListedColormap(color_list)
cmap.set_over(color_list[-1])
cmap.set_under(color_list[0])
invcmap = mpl.colors.ListedColormap(inv_color_list)
invcmap.set_over(inv_color_list[-1])
invcmap.set_under(inv_color_list[0])

__all__ = ['generateChisquareMap', 'generateCorrelationPlot', 'generateWalkPlot']

def _make_axes_grid(no_variables, padding=0, cbar_size=0.5, axis_padding=0.5, cbar=True):
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
        Tuple containing the figure, a 2D-array of axes and the colorbar axis."""

    # Convert to inches.
    padding, cbar_size, axis_padding = (padding * 0.393700787,
                                        cbar_size * 0.393700787,
                                        axis_padding * 0.393700787)
    if not cbar:
        cbar_size = 0

    # Generate the figure, convert padding to percentages.
    fig = plt.figure()
    padding = 1

    axis_size_left = (fig.get_figwidth()-padding - 0*(no_variables + 1) * padding) / no_variables
    axis_size_up = (fig.get_figheight()-padding - 0*(no_variables + 1) * padding) / no_variables

    cbar_size = cbar_size / fig.get_figwidth()
    left_padding = padding * 0.5 / fig.get_figwidth()
    left_axis_padding = axis_padding / fig.get_figwidth()
    up_padding = padding * 0.5 / fig.get_figheight()
    up_axis_padding = 0*axis_padding / fig.get_figheight()
    axis_size_left = axis_size_left / fig.get_figwidth()
    axis_size_up = axis_size_up / fig.get_figheight()

    # Pre-allocate a 2D-array to hold the axes.
    axes = np.array([[None for _ in range(no_variables)] for _ in range(no_variables)],
                    dtype='object')

    for i, I in zip(range(no_variables), reversed(range(no_variables))):
        for j in reversed(range(no_variables)):
            # Only create axes on the lower triangle.
            if I + j < no_variables:
                # Share the x-axis with the plot on the diagonal,
                # directly above the plot.
                sharex = axes[j, j] if i != j else None
                # Share the y-axis among the 2D maps along one row,
                # but not the plot on the diagonal!
                sharey = axes[i, i-1] if (i != j and i-1 != j) else None
                # Determine the place and size of the axes
                left_edge = j * axis_size_left + left_padding
                bottom_edge = I * axis_size_up + up_padding
                if j > 0:
                    left_edge += j * left_axis_padding
                if I > 0:
                    bottom_edge += I * up_axis_padding

                a = plt.axes([left_edge, bottom_edge, axis_size_left, axis_size_up],
                             sharex=sharex, sharey=sharey)
                plt.setp(a.xaxis.get_majorticklabels(), rotation=45)
                plt.setp(a.yaxis.get_majorticklabels(), rotation=45)
            else:
                a = None
            if i == j:
                a.yaxis.tick_right()
                a.yaxis.set_label_position('right')
            axes[i, j] = a

    axes = np.array(axes)
    for a in axes[:-1, :].flatten():
        if a is not None:
            plt.setp(a.get_xticklabels(), visible=False)
    for a in axes[:, 1:].flatten():
        if a is not None:
            plt.setp(a.get_yticklabels(), visible=False)
    left_edge = no_variables*(axis_size_left+left_axis_padding)+left_padding
    bottom_edge = up_padding
    width = cbar_size

    height = axis_size_up * len(axes) + up_padding * (len(axes) - 1)

    cbar_width = axis_size_left * 0.1
    if cbar:
        cbar = plt.axes([1-cbar_width-padding*0.5/fig.get_figwidth(), padding*0.5/fig.get_figheight()+axis_size_up*1.5, cbar_width, axis_size_up*(no_variables-1)-axis_size_up*0.5])
        plt.setp(cbar.get_xticklabels(), visible=False)
        plt.setp(cbar.get_yticklabels(), visible=False)
    else:
        cbar = None
    return fig, axes, cbar

def generateChisquareMap(fitter, filter=None, method='chisquare', resolution_diag=15, resolution_map=15, fit_kws={}, source=False, model=True):
    """Generates a correlation map for either the chisquare or the MLE method.
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

    title = '{}\n${}_{{-{}}}^{{+{}}}$'
    title_e = '{}\n$({}_{{-{}}}^{{+{}}})e{}$'

    try:
        orig_value = fitter.chisqr
    except AttributeError:
        fitter.fit(**fit_kws)
        orig_value = fitter.chisqr
    if method.lower().startswith('llh'):
        orig_value = fitter.llh_result
    result = copy.deepcopy(fitter.result)
    orig_params = copy.deepcopy(fitter.lmpars)

    ranges = {}

    param_names = []
    no_params = 0
    for p in orig_params:
        if orig_params[p].vary and (filter is None or any([f in p for f in filter])):
            no_params += 1
            param_names.append(p)
    fig, axes, cbar = _make_axes_grid(no_params, axis_padding=0, cbar=no_params > 1)

    split_names = [name.split('___') for name in param_names]
    sources = [name[0] for name in split_names]
    models = [name[1] for name in split_names]
    var_names = [name[2] for name in split_names]
    to_be_combined = [var_names]
    if model:
        to_be_combined.insert(0, models)
    if source:
        to_be_combined.insert(0, sources)

    var_names = [' '.join(tbc) for tbc in zip(*to_be_combined)]

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
        if i == no_params-1:
            if method.lower().startswith('chisquare'):
                ax.set_ylabel(r'$\Delta\chi^2$')
            else:
                ax.set_ylabel(r'$\Delta\mathcal{L}$')
                fit_kws['llh_selected'] = True

        # Select starting point to determine error widths.
        value = orig_params[param_names[i]].value
        stderr = orig_params[param_names[i]].stderr
        stderr = stderr if stderr is not None else 0.01 * np.abs(value)
        stderr = stderr if stderr != 0 else 0.01 * np.abs(value)

        right = value + stderr
        left = value - stderr
        params[param_names[i]].vary = False

        ranges[param_names[i]]['left_val'] = 3*left - 2*value
        ranges[param_names[i]]['right_val'] = 3*right - 2*value
        value_range = np.linspace(3*left - 2*value, right*3 - 2*value, resolution_diag)
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
        ax.plot(value_range, chisquare, color='k')

        c = '#0093e6'
        ax.axvline(right, ls="dashed", color=c)
        ax.axvline(left, ls="dashed", color=c)
        ax.axvline(value, ls="dashed", color=c)
        up = '{:.2ug}'.format(u.ufloat(value, stderr))
        down = '{:.2ug}'.format(u.ufloat(value, stderr))
        val = up.split('+/-')[0].split('(')[-1]
        r = up.split('+/-')[1].split(')')[0]
        l = down.split('+/-')[1].split(')')[0]
        if 'e' in up or 'e' in down:
            ex = up.split('e')[-1]
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
        right = ranges[x_name]['right_val']
        left = ranges[x_name]['left_val']
        x_range = np.linspace(left, right, resolution_map)

        right = ranges[y_name]['right_val']
        left = ranges[y_name]['left_val']
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
                Z[k, l] = (fitter.llh_result - orig_value)*2
            else:
                Z[k, l] = fitter.chisqr - orig_value

        Z = -Z
        bounds = []
        for bound in [0.997300204, 0.954499736, 0.682689492]:
            chifunc = lambda x: chi2.cdf(x, 1) - bound # Calculate 1 sigma boundary
            bounds.append(-optimize.root(chifunc, 1).x[0])
        bounds.append(0)
        bounds = np.array(bounds)
        norm = mpl.colors.BoundaryNorm(bounds, invcmap.N)
        contourset = ax.contourf(X, Y, Z, bounds, cmap=invcmap, norm=norm)
        fitter.lmpars = copy.deepcopy(orig_params)
    try:
        cbar = plt.colorbar(contourset, cax=cbar, orientation='vertical')
        cbar.ax.yaxis.set_ticks([-7.5, -4.5, -1.5])
        cbar.ax.set_yticklabels([r'3$\sigma$', r'2$\sigma$', r'1$\sigma$'])
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

def generateCorrelationPlot(filename, filter=None, bins=None, selection=(0, 100), source=False, model=True):
    """Given the random walk data, creates a triangle plot: distribution of
    a single parameter on the diagonal axes, 2D contour plots with 1, 2 and
    3 sigma contours on the off-diagonal. The 1-sigma limits based on the
    percentile method are also indicated, as well as added to the title.

    Parameters
    ----------
    filename: string
        Filename for the h5 file containing the data from the walk.
    filter: list of str, optional
        If supplied, only this list of columns is used for the plot.
    bins: int or list of int, optional
        If supplied, use this number of bins for the plotting.

    Returns
    -------
    figure
        Returns the MatPlotLib figure created."""

    reader = SATLASHDFBackend(filename)
    var_names = list(reader.labels)
    split_names = [name.split('___') for name in var_names]
    sources = [name[0]+'\n' for name in split_names]
    models = [name[1] for name in split_names]
    var_names = [name[2] for name in split_names]
    to_be_combined = [var_names]
    if model:
        to_be_combined.insert(0, models)
    if source:
        to_be_combined.insert(0, sources)

    var_names = [' '.join(tbc) for tbc in zip(*to_be_combined)]

    data = reader.get_chain(flat=False)
    dataset_length = data.shape[0]
    first, last = int(np.floor(dataset_length/100*selection[0])), int(np.ceil(dataset_length/100*selection[1]))
    data = data[first:last, :, :]
    data = data.reshape(-1, data.shape[-1])

    if filter is not None:
        filter = [c for f in filter for c in var_names if f in c]
    else:
        filter = var_names
    with tqdm.tqdm(total=len(filter)+(len(filter)**2-len(filter))/2, leave=True) as pbar:
        fig, axes, cbar = _make_axes_grid(len(filter), axis_padding=0)

        metadata = {}
        if not isinstance(bins, list):
            bins = [bins for _ in filter]
        for i, val in enumerate(filter):
            pbar.set_description(val)
            ax = axes[i, i]
            bin_index = i
            i = var_names.index(val)
            x = data[:, i]

            if bins[bin_index] is None:
                width = 3.5*np.std(x)/x.size**(1/3) #Scott's rule for binwidth
                bins[bin_index] = np.arange(x.min(), x.max()+width, width)
            try:
                n, b, p, = ax.hist(x, int(bins[bin_index]), histtype='step', color='k')
            except TypeError:
                bins[bin_index] = 50
                n, b, p, = ax.hist(x, int(bins[bin_index]), histtype='step', color='k')
            # center = n.argmax()
            # q50 = (b[center] + b[center+1])/2
            q = [15.87, 50, 84.13]
            q16, q50, q84 = np.percentile(x, q)
            metadata[val] = {'bins': bins[bin_index], 'min': x.min(), 'max': x.max()}


            title = '{}\n${}_{{-{}}}^{{+{}}}$'
            title_e = '{}\n$({}_{{-{}}}^{{+{}}})e{}$'
            up = '{:.2ug}'.format(u.ufloat(q50, np.abs(q84-q50)))
            down = '{:.2ug}'.format(u.ufloat(q50, np.abs(q50-q16)))
            param_val = up.split('+/-')[0].split('(')[-1]
            r = up.split('+/-')[1].split(')')[0]
            l = down.split('+/-')[1].split(')')[0]
            if 'e' in up or 'e' in down:
                ex = up.split('e')[-1]
                ax.set_title(title_e.format(val, param_val, l, r, ex))
            else:
                ax.set_title(title.format(val, param_val, l, r))

            qvalues = [q16, q50, q84]
            c = '#0093e6'
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=c)
            ax.set_yticks([])
            ax.set_yticklabels([])
            pbar.update(1)

        for i, j in zip(*np.tril_indices_from(axes, -1)):
            x_name = filter[j]
            y_name = filter[i]
            pbar.set_description(', '.join([x_name, y_name]))
            ax = axes[i, j]
            if j == 0:
                ax.set_ylabel(filter[i])
            if i == len(filter) - 1:
                ax.set_xlabel(filter[j])
            j = var_names.index(x_name)
            i = var_names.index(y_name)
            x = data[:, j]
            y = data[:, i]
            x_min, x_max, x_bins = metadata[x_name]['min'], metadata[x_name]['max'], metadata[x_name]['bins']
            y_min, y_max, y_bins = metadata[y_name]['min'], metadata[y_name]['max'], metadata[y_name]['bins']
            X = np.linspace(x_min, x_max, x_bins + 1)
            Y = np.linspace(y_min, y_max, y_bins + 1)
            H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=(X, Y),
                                     weights=None)
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

            contourset = ax.contourf(X1, Y1, H.T, bounds, cmap=invcmap, norm=norm)
            pbar.update(1)
        try:
            cbar = plt.colorbar(contourset, cax=cbar, orientation='vertical')
            cbar.ax.yaxis.set_ticks([0, 1/6, 0.5, 5/6])
            cbar.ax.set_yticklabels(['', r'3$\sigma$', r'2$\sigma$', r'1$\sigma$'])
        except:
            cbar = None
    return fig, axes, cbar

def generateWalkPlot(filename, filter=None, selection=(0, 100), walkers=20, source=False, model=True):
    """Given the random walk data, the random walk for the selected parameters
    is plotted.

    Parameters
    ----------
    filename: string
        Filename for the h5 file containing the data from the walk.
    filter: list of str, optional
        If supplied, only this list of parameters is used for the plot.

    Returns
    -------
    figure
        Returns the MatPlotLib figure created."""

    reader = SATLASHDFBackend(filename)
    var_names = reader.labels
    split_names = [name.split('___') for name in var_names]
    sources = [name[0] for name in split_names]
    models = [name[1] for name in split_names]
    var_names = [name[2] for name in split_names]
    to_be_combined = [var_names]
    if model:
        to_be_combined.insert(0, models)
    if source:
        to_be_combined.insert(0, sources)

    var_names = [' '.join(tbc) for tbc in zip(*to_be_combined)]

    data = reader.get_chain(flat=False)
    dataset_length = data.shape[0]
    first, last = int(np.floor(dataset_length/100*selection[0])), int(np.ceil(dataset_length/100*selection[1]))
    data = data[first:last, :, :]
    # data = data.reshape(-1, data.shape[-1])

    if filter is not None:
        filter = [c for f in filter for c in var_names if f in c]
    else:
        filter = var_names
    with tqdm.tqdm(total=len(filter), leave=True) as pbar:
        fig, axes = plt.subplots(len(filter), 1, sharex=True)

        for i, (val, ax) in enumerate(zip(filter, axes)):
            pbar.set_description(val)
            i = var_names.index(val)
            x = data[:, :, i]
            q50 = np.percentile(x, [50.0])
            ax.plot(range(first, last), x, alpha=0.3, color='gray')
            ax.set_ylabel(val)
            ax.axhline(q50, color='k')
            pbar.update(1)
        ax.set_xlabel('Step')
    pbar.close()
    return fig, axes
