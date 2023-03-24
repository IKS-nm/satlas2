import sys

import matplotlib
from matplotlib import gridspec

matplotlib.use('Qt5Agg')

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5 import Qt, QtWidgets

sys.path.insert(0, '..\src')

import satlas2


class CustomLlhFitter(satlas2.Fitter):
    def customLlh(self):
        attr = 'bunches' if hasattr(self.sources[0][1],
                                    'bunches') else 'bunches_noplot'
        try:
            bunches = self.bunches
        except:
            bunches = self.getSourceAttr(attr)
            self.bunches = bunches
        try:
            data_counts = self.data_counts
        except:
            data_counts = self.temp_y * bunches
            self.data_counts = data_counts
        model_rates = self.f()
        model_counts = model_rates * bunches
        returnvalue = data_counts * np.log(model_counts) - model_counts
        returnvalue[model_counts <= 0] = -np.inf
        priors = self.gaussianPriorResid()
        if len(priors) > 1:
            priors = -0.5 * priors * priors
            returnvalue = np.append(returnvalue, priors)
        return returnvalue


def modifiedSqrt(y, bunches=None):
    yerr = np.sqrt(y * bunches) / bunches
    yerr[y <= 0] = 1 / bunches[y <= 0]
    return yerr


def update_errorbar(errobj, x, y, xerr=None, yerr=None):
    ln, caps, bars = errobj

    if len(bars) == 2:
        assert xerr is not None and yerr is not None, "Your errorbar object has 2 dimension of error bars defined. You must provide xerr and yerr."
        barsx, barsy = bars  # bars always exist (?)
        try:  # caps are optional
            errx_top, errx_bot, erry_top, erry_bot = caps
        except ValueError:  # in case there is no caps
            pass

    elif len(bars) == 1:
        assert (xerr is     None and yerr is not None) or\
               (xerr is not None and yerr is     None),  \
               "Your errorbar object has 1 dimension of error bars defined. You must provide xerr or yerr."

        if xerr is not None:
            barsx, = bars  # bars always exist (?)
            try:
                errx_top, errx_bot = caps
            except ValueError:  # in case there is no caps
                pass
        else:
            barsy, = bars  # bars always exist (?)
            try:
                erry_top, erry_bot = caps
            except ValueError:  # in case there is no caps
                pass

    ln.set_data(x, y)

    try:
        errx_top.set_xdata(x + xerr)
        errx_bot.set_xdata(x - xerr)
        errx_top.set_ydata(y)
        errx_bot.set_ydata(y)
    except NameError:
        pass
    try:
        barsx.set_segments([
            np.array([[xt, y], [xb, y]])
            for xt, xb, y in zip(x + xerr, x - xerr, y)
        ])
    except NameError:
        pass

    try:
        erry_top.set_xdata(x)
        erry_bot.set_xdata(x)
        erry_top.set_ydata(y + yerr)
        erry_bot.set_ydata(y - yerr)
    except NameError:
        pass
    try:
        barsy.set_segments([
            np.array([[x, yt], [x, yb]])
            for x, yt, yb in zip(x, y + yerr, y - yerr)
        ])
    except NameError:
        pass


class QHSeparationLine(QtWidgets.QFrame):
    '''
  a horizontal separation line\n
  '''
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Minimum)
        return


class QVSeparationLine(QtWidgets.QFrame):
    '''
  a vertical separation line\n
  '''
    def __init__(self):
        super().__init__()
        self.setFixedWidth(20)
        self.setMinimumHeight(1)
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
                           QtWidgets.QSizePolicy.Preferred)
        return


class FitterPlotter(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height),
                          dpi=dpi,
                          constrained_layout=True)
        self.errorbarPlots = []
        self.initPlots = []
        self.fitPlots = []
        self.axes = []
        super().__init__(self.fig)

    def drawData(self, f, boundaries, fit=True, fit_kwargs={}):
        self.drawInit(f, boundaries)
        if fit:
            self.drawFits(f, boundaries, fit_kwargs)

    def drawFits(self, f, boundaries, fit_kwargs):
        try:
            f.fit(**fit_kwargs)
            x = f.sources[0][1].x

            left = np.append(-np.inf, boundaries)
            right = np.append(boundaries, np.inf)
            for i, (l, r) in enumerate(zip(left, right)):
                mask = np.bitwise_and.reduce([x >= l, x <= r])
                X = x[mask]

                ax = self.axes[i]
                fitplot = self.fitPlots[i]

                plot_x = np.arange(X.min(), X.max() + 1)
                plot_y = f.sources[0][1].evaluate(plot_x)
                fitplot.set_data(plot_x, plot_y)
                ax.relim()
                ax.autoscale()
                ax.autoscale_view()
        except:
            pass

    def drawInit(self, f, boundaries):
        if len(self.initPlots) != len(boundaries) + 1:
            self.fig.clear()
            gs = gridspec.GridSpec(nrows=1,
                                   ncols=len(boundaries) + 1,
                                   figure=self.fig)
            self.errorbarPlots = []
            self.initPlots = []
            self.fitPlots = []
            self.axes = []
            for i in range(len(boundaries) + 1):
                ax = self.fig.add_subplot(gs[0, i])
                ax.label_outer()
                if i == 0:
                    ax.set_ylabel('Counts [-]')
                ax.set_xlabel('Frequency [MHz]')
                erb = ax.errorbar([], [],
                                  yerr=[],
                                  drawstyle='steps-mid',
                                  label='Data')
                self.errorbarPlots.append(erb)
                line, = ax.plot([], [], label='Init')
                self.initPlots.append(line)
                line, = ax.plot([], [], label='Fit')
                self.fitPlots.append(line)
                self.axes.append(ax)
        x = f.sources[0][1].x
        y = f.sources[0][1].y
        yerr = f.sources[0][1].yerr()

        left = np.append(-np.inf, boundaries)
        right = np.append(boundaries, np.inf)
        for i, (l, r) in enumerate(zip(left, right)):
            mask = np.bitwise_and.reduce([x >= l, x <= r])
            X = x[mask]
            Y = y[mask]
            YERR = yerr[mask]

            dataerb = self.errorbarPlots[i]
            ax = self.axes[i]
            initplot = self.initPlots[i]
            fitplot = self.fitPlots[i]

            update_errorbar(dataerb, X, Y, yerr=YERR)

            plot_x = np.arange(X.min(), X.max() + 1)
            plot_y = f.sources[0][1].evaluate(plot_x)
            initplot.set_data(plot_x, plot_y)
            fitplot.set_data(plot_x, plot_y)
            ax.relim()
            ax.autoscale()
            ax.autoscale_view()


class SimulatorPlotter(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height),
                          dpi=dpi,
                          constrained_layout=True)
        self.line_rates = []
        self.ax_rates = []
        super().__init__(self.fig)

    def drawSingle(self, f, boundaries):
        if len(self.line_rates) != len(boundaries) + 1:
            self.fig.clear()
            gs = gridspec.GridSpec(nrows=1,
                                   ncols=len(boundaries) + 1,
                                   figure=self.fig)
            self.line_rates = []
            self.ax_rates = []
        x = f.sources[0][1].x
        left = np.append(-np.inf, boundaries)
        right = np.append(boundaries, np.inf)
        for i, (l, r) in enumerate(zip(left, right)):
            mask = np.bitwise_and.reduce([x >= l, x <= r])
            X = x[mask]

            plot_x = np.arange(X.min(), X.max() + 1)
            plot_y = f.sources[0][1].evaluate(plot_x)
            try:
                self.line_rates[i].set_data(plot_x, plot_y)
                self.ax_rates[i].relim()
                self.ax_rates[i].autoscale()
                self.ax_rates[i].autoscale_view()
            except:
                ax_r = self.fig.add_subplot(gs[0, i])

                ax_r.set_ylabel('Rate [cts/bunch]')

                ax_r.set_xlabel('Frequency [MHz]')
                line_r, = ax_r.plot(plot_x,
                                    f.sources[0][1].evaluate(plot_x),
                                    label='Init')
                self.line_rates.append(line_r)
                self.ax_rates.append(ax_r)

                ax_r.label_outer()

    def drawData(self, f, boundaries):
        if hasattr(f.sources[0][1], 'bunches'):
            self.drawTriple(f, boundaries)
        else:
            self.drawDouble(f, boundaries)
        self.line_rates = []
        self.ax_rates = []

    def drawDouble(self, f, boundaries):
        self.fig.clear()
        x = f.sources[0][1].x
        y = f.sources[0][1].y
        yerr = f.sources[0][1].yerr()
        bunches = f.sources[0][1].bunches_noplot
        gs = gridspec.GridSpec(nrows=2,
                               ncols=len(boundaries) + 1,
                               figure=self.fig)
        left = np.append(-np.inf, boundaries)
        right = np.append(boundaries, np.inf)
        ax_rates = []
        ax_counts = []
        for i, (l, r) in enumerate(zip(left, right)):
            mask = np.bitwise_and.reduce([x >= l, x <= r])
            X = x[mask]
            Y = y[mask]
            BUNCHES = bunches[mask]
            YERR = yerr[mask]

            try:
                ax_c = self.fig.add_subplot(gs[0, i], sharey=ax_counts[0])
            except:
                ax_c = self.fig.add_subplot(gs[0, i])
            ax_counts.append(ax_c)

            try:
                ax_r = self.fig.add_subplot(gs[1, i],
                                            sharex=ax_c,
                                            sharey=ax_rates[0])
            except:
                ax_r = self.fig.add_subplot(gs[1, i], sharex=ax_c)
            ax_rates.append(ax_r)

            ax_c.plot(X, Y * BUNCHES, drawstyle='steps-mid', label='Data')
            ax_r.errorbar(X, Y, yerr=YERR, drawstyle='steps-mid', label='Data')

            ax_r.set_ylabel('Rate [cts/bunch]')
            ax_c.set_ylabel('Raw counts')

            ax_r.set_xlabel('Frequency [MHz]')

            plot_x = np.arange(X.min(), X.max() + 1)
            ax_r.plot(plot_x, f.sources[0][1].evaluate(plot_x), label='Init')

            f.fit(llh=True, llh_method='custom')
            ax_r.plot(plot_x, f.sources[0][1].evaluate(plot_x), label='Fit')

            if i == 0:
                ax_r.legend(loc=0)

            ax_c.label_outer()
            ax_r.label_outer()

    def drawTriple(self, f, boundaries):
        self.fig.clear()
        x = f.sources[0][1].x
        y = f.sources[0][1].y
        yerr = f.sources[0][1].yerr()
        bunches = f.sources[0][1].bunches

        gs = gridspec.GridSpec(nrows=3,
                               ncols=len(boundaries) + 1,
                               figure=self.fig)
        left = np.append(-np.inf, boundaries)
        right = np.append(boundaries, np.inf)
        ax_events = []
        ax_counts = []
        ax_rates = []
        for i, (l, r) in enumerate(zip(left, right)):
            mask = np.bitwise_and.reduce([x >= l, x <= r])
            X = x[mask]
            Y = y[mask]
            BUNCHES = bunches[mask]
            YERR = yerr[mask]

            try:
                ax_e = self.fig.add_subplot(gs[0, i], sharey=ax_events[0])
            except:
                ax_e = self.fig.add_subplot(gs[0, i])
            ax_events.append(ax_e)

            try:
                ax_c = self.fig.add_subplot(gs[1, i],
                                            sharey=ax_counts[0],
                                            sharex=ax_e)
            except:
                ax_c = self.fig.add_subplot(gs[1, i], sharex=ax_e)
            ax_counts.append(ax_c)

            try:
                ax_r = self.fig.add_subplot(gs[2, i],
                                            sharex=ax_e,
                                            sharey=ax_rates[0])
            except:
                ax_r = self.fig.add_subplot(gs[2, i], sharex=ax_e)
            ax_rates.append(ax_r)

            ax_e.plot(X, BUNCHES, drawstyle='steps-mid')
            ax_c.plot(X, Y * BUNCHES, drawstyle='steps-mid', label='Data')
            ax_r.errorbar(X, Y, yerr=YERR, drawstyle='steps-mid', label='Data')

            ax_r.set_ylabel('Rate [cts/bunch]')
            ax_e.set_ylabel('Number of bunches [-]')
            ax_c.set_ylabel('Raw counts')

            ax_r.set_xlabel('Frequency [MHz]')

            plot_x = np.arange(X.min(), X.max() + 1)
            ax_r.plot(plot_x, f.sources[0][1].evaluate(plot_x))

            f.fit(llh=True, llh_method='custom')
            ax_r.plot(plot_x, f.sources[0][1].evaluate(plot_x), label='Fit')

            if i == 0:
                ax_r.legend(loc=0)

            ax_e.label_outer()
            ax_c.label_outer()
            ax_r.label_outer()


class ParameterWidget(QtWidgets.QWidget):
    sigChanged = Qt.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QtWidgets.QVBoxLayout()
        self.SpinSpinBox = QtWidgets.QDoubleSpinBox()
        self.SpinSpinBox.setMinimum(0)
        self.SpinSpinBox.setValue(0)
        self.SpinSpinBox.setSingleStep(0.5)

        # J = [0, 1]
        self.J1SpinBox = QtWidgets.QDoubleSpinBox()
        self.J1SpinBox.setMinimum(0)
        self.J1SpinBox.setValue(0)
        self.J1SpinBox.setSingleStep(0.5)

        self.J2SpinBox = QtWidgets.QDoubleSpinBox()
        self.J2SpinBox.setMinimum(0)
        self.J2SpinBox.setValue(1)
        self.J2SpinBox.setSingleStep(0.5)

        # A = [0, 50]
        self.A1SpinBox = QtWidgets.QDoubleSpinBox()
        self.A1SpinBox.setValue(0)
        self.A1SpinBox.setMinimum(-90000)
        self.A1SpinBox.setMaximum(90000)

        self.A2SpinBox = QtWidgets.QDoubleSpinBox()
        self.A2SpinBox.setValue(50)
        self.A2SpinBox.setMinimum(-90000)
        self.A2SpinBox.setMaximum(90000)
        # B = [0, 0]
        self.B1SpinBox = QtWidgets.QDoubleSpinBox()
        self.B1SpinBox.setValue(0)
        self.B1SpinBox.setMinimum(-90000)
        self.B1SpinBox.setMaximum(90000)

        self.B2SpinBox = QtWidgets.QDoubleSpinBox()
        self.B2SpinBox.setValue(0)
        self.B2SpinBox.setMinimum(-90000)
        self.B2SpinBox.setMaximum(90000)
        # C = [0, 0]
        self.C1SpinBox = QtWidgets.QDoubleSpinBox()
        self.C1SpinBox.setValue(0)
        self.C1SpinBox.setMinimum(-90000)
        self.C1SpinBox.setMaximum(90000)

        self.C2SpinBox = QtWidgets.QDoubleSpinBox()
        self.C2SpinBox.setValue(0)
        self.C2SpinBox.setMinimum(-90000)
        self.C2SpinBox.setMaximum(90000)
        # FWHMG = 135 / 4
        self.FWHMGSpinBox = QtWidgets.QDoubleSpinBox()
        self.FWHMGSpinBox.setMinimum(0.001)
        self.FWHMGSpinBox.setMaximum(90000)
        self.FWHMGSpinBox.setValue(100)
        # FWHML = 101 / 25
        self.FWHMLSpinBox = QtWidgets.QDoubleSpinBox()
        self.FWHMLSpinBox.setMinimum(0.001)
        self.FWHMLSpinBox.setMaximum(90000)
        self.FWHMLSpinBox.setValue(100)
        # centroid = 0
        self.CentroidSpinBox = QtWidgets.QDoubleSpinBox()
        self.CentroidSpinBox.setMaximum(90000)
        self.CentroidSpinBox.setMinimum(-90000)
        self.CentroidSpinBox.setValue(0)
        # bkg = 1
        self.BkgSpinBox = QtWidgets.QDoubleSpinBox()
        self.BkgSpinBox.setMinimum(0)
        self.BkgSpinBox.setMaximum(90000)
        self.BkgSpinBox.setValue(1)
        self.BkgSpinBox.setDecimals(6)
        # scale = 1
        self.ScaleSpinBox = QtWidgets.QDoubleSpinBox()
        self.ScaleSpinBox.setMinimum(0.000001)
        self.ScaleSpinBox.setMaximum(90000)
        self.ScaleSpinBox.setValue(1)
        self.ScaleSpinBox.setDecimals(6)

        self.SamplesSpinBox = QtWidgets.QSpinBox()
        self.SamplesSpinBox.setMinimum(1)
        self.SamplesSpinBox.setMaximum(10000)
        self.SamplesSpinBox.setValue(1000)

        self.SamplesNoiseSpinBox = QtWidgets.QSpinBox()
        self.SamplesNoiseSpinBox.setMinimum(1)
        self.SamplesNoiseSpinBox.setMaximum(10000)
        self.SamplesNoiseSpinBox.setValue(5)

        self.SampleStepSpinbox = QtWidgets.QDoubleSpinBox()
        self.SampleStepSpinbox.setMinimum(1)
        self.SampleStepSpinbox.setMaximum(90000)
        self.SampleStepSpinbox.setValue(25)

        self.LabelSpin = QtWidgets.QLabel('Spin')
        self.LabelJ1 = QtWidgets.QLabel('J1')
        self.LabelJ2 = QtWidgets.QLabel('J2')
        self.LabelAl = QtWidgets.QLabel('Al')
        self.LabelAlResult = QtWidgets.QLabel('')
        self.LabelAu = QtWidgets.QLabel('Au')
        self.LabelAuResult = QtWidgets.QLabel('')
        self.LabelBl = QtWidgets.QLabel('Bl')
        self.LabelBlResult = QtWidgets.QLabel('')
        self.LabelBu = QtWidgets.QLabel('Bu')
        self.LabelBuResult = QtWidgets.QLabel('')
        self.LabelCl = QtWidgets.QLabel('Cl')
        self.LabelClResult = QtWidgets.QLabel('')
        self.LabelCu = QtWidgets.QLabel('Cu')
        self.LabelCuResult = QtWidgets.QLabel('')
        self.LabelFWHMG = QtWidgets.QLabel('FWHMG')
        self.LabelFWHMGResult = QtWidgets.QLabel('')
        self.LabelFWHML = QtWidgets.QLabel('FWHML')
        self.LabelFWHMLResult = QtWidgets.QLabel('')
        self.LabelCentroid = QtWidgets.QLabel('Centroid')
        self.LabelCentroidResult = QtWidgets.QLabel('')
        self.LabelBackground = QtWidgets.QLabel('Background')
        self.LabelBackgroundResult = QtWidgets.QLabel('')
        self.LabelScale = QtWidgets.QLabel('Scale')
        self.LabelScaleResult = QtWidgets.QLabel('')
        self.LabelSamples = QtWidgets.QLabel('Samples')
        self.LabelSamplesNoise = QtWidgets.QLabel('Sample noise')
        self.LabelSampleStep = QtWidgets.QLabel('Stepsize')

        self.LabelSampleMode = QtWidgets.QLabel('Sampling mode')

        self.samplingConstant = QtWidgets.QRadioButton('Constant')
        self.samplingGaussian = QtWidgets.QRadioButton('Gaussian')
        self.samplingPoisson = QtWidgets.QRadioButton('Poisson')
        self.buttonGroup = QtWidgets.QButtonGroup()
        self.buttonGroup.addButton(self.samplingConstant, 1)
        self.buttonGroup.addButton(self.samplingGaussian, 2)
        self.buttonGroup.addButton(self.samplingPoisson, 3)
        self.samplingModeButtons = [
            self.samplingConstant, self.samplingGaussian, self.samplingPoisson
        ]

        sampling_groupbox = QtWidgets.QGroupBox('Simulation info')
        sampling_grid = QtWidgets.QGridLayout()
        sampling_grid.addWidget(self.LabelSamples, 0, 0)
        sampling_grid.addWidget(self.SamplesSpinBox, 0, 1)
        sampling_grid.addWidget(self.LabelSamplesNoise, 1, 0)
        sampling_grid.addWidget(self.SamplesNoiseSpinBox, 1, 1)
        sampling_grid.addWidget(self.LabelSampleMode, 2, 0)
        temp_layout = QtWidgets.QVBoxLayout()
        for w in self.samplingModeButtons:
            temp_layout.addWidget(w)
        sampling_grid.addLayout(temp_layout, 2, 1)
        sampling_groupbox.setLayout(sampling_grid)
        layout.addWidget(sampling_groupbox)

        nuclear_groupbox = QtWidgets.QGroupBox('Nuclear info')
        nuclear_grid = QtWidgets.QGridLayout()
        labels = [
            self.LabelSpin, self.LabelJ1, self.LabelJ2, self.LabelAl,
            self.LabelAu, self.LabelBl, self.LabelBu, self.LabelCl,
            self.LabelCu, self.LabelFWHMG, self.LabelFWHML, self.LabelCentroid,
            self.LabelBackground, self.LabelScale
        ]
        for i, label in enumerate(labels):
            nuclear_grid.addWidget(label, i, 0)
        widgets = [
            self.SpinSpinBox, self.J1SpinBox, self.J2SpinBox, self.A1SpinBox,
            self.A2SpinBox, self.B1SpinBox, self.B2SpinBox, self.C1SpinBox,
            self.C2SpinBox, self.FWHMGSpinBox, self.FWHMLSpinBox,
            self.CentroidSpinBox, self.BkgSpinBox, self.ScaleSpinBox
        ]
        for i, widget in enumerate(widgets):
            nuclear_grid.addWidget(widget, i, 1)
        extralabels = [
            self.LabelAlResult, self.LabelAuResult, self.LabelBlResult,
            self.LabelBuResult, self.LabelClResult, self.LabelCuResult,
            self.LabelFWHMGResult, self.LabelFWHMLResult,
            self.LabelCentroidResult, self.LabelBackgroundResult,
            self.LabelScaleResult
        ]
        for i, label in enumerate(extralabels):
            nuclear_grid.addWidget(label, i + 3, 2)
        nuclear_groupbox.setLayout(nuclear_grid)
        layout.addWidget(nuclear_groupbox)

        self.samplingConstant.toggle()
        self.setLayout(layout)
        self.SpinSpinBox.valueChanged.connect(self.sigChanged.emit)
        self.J1SpinBox.valueChanged.connect(self.sigChanged.emit)
        self.J2SpinBox.valueChanged.connect(self.sigChanged.emit)
        self.A1SpinBox.valueChanged.connect(self.sigChanged.emit)
        self.A2SpinBox.valueChanged.connect(self.sigChanged.emit)
        self.B1SpinBox.valueChanged.connect(self.sigChanged.emit)
        self.B2SpinBox.valueChanged.connect(self.sigChanged.emit)
        self.C1SpinBox.valueChanged.connect(self.sigChanged.emit)
        self.C2SpinBox.valueChanged.connect(self.sigChanged.emit)
        self.FWHMGSpinBox.valueChanged.connect(self.sigChanged.emit)
        self.FWHMLSpinBox.valueChanged.connect(self.sigChanged.emit)
        self.CentroidSpinBox.valueChanged.connect(self.sigChanged.emit)
        self.BkgSpinBox.valueChanged.connect(self.sigChanged.emit)
        self.ScaleSpinBox.valueChanged.connect(self.sigChanged.emit)

    def updateLabels(self, f):
        params = f.sources[0][1].models[0][1].params
        self.LabelAlResult.setText(params['Al'].representation())
        self.LabelAuResult.setText(params['Au'].representation())
        self.LabelBlResult.setText(params['Bl'].representation())
        self.LabelBuResult.setText(params['Bu'].representation())
        self.LabelClResult.setText(params['Cl'].representation())
        self.LabelCuResult.setText(params['Cu'].representation())
        self.LabelFWHMGResult.setText(params['FWHMG'].representation())
        self.LabelFWHMLResult.setText(params['FWHML'].representation())
        self.LabelCentroidResult.setText(params['centroid'].representation())
        self.LabelScaleResult.setText(params['scale'].representation())
        params = f.sources[0][1].models[1][1].params
        self.LabelBackgroundResult.setText(params['p0'].representation())

    def getSampleMode(self):
        mapping = {1: 'constant', 2: 'gaussian', 3: 'poisson'}
        return mapping[self.buttonGroup.checkedId()]

    def getSamples(self):
        return int(self.SamplesSpinBox.value()), int(
            self.SamplesNoiseSpinBox.value()), float(
                self.SampleStepSpinbox.value())

    def getParameters(self):
        I = float(self.SpinSpinBox.value())
        J = [float(self.J1SpinBox.value()), float(self.J2SpinBox.value())]
        A = [float(self.A1SpinBox.value()), float(self.A2SpinBox.value())]
        B = [float(self.B1SpinBox.value()), float(self.B2SpinBox.value())]
        C = [float(self.C1SpinBox.value()), float(self.C2SpinBox.value())]
        FWHMG = float(self.FWHMGSpinBox.value())
        FWHML = float(self.FWHMLSpinBox.value())
        Centroid = float(self.CentroidSpinBox.value())
        Background = float(self.BkgSpinBox.value())
        Scale = float(self.ScaleSpinBox.value())
        returndict = {
            'I': I,
            'J': J,
            'A': A,
            'B': B,
            'C': C,
            'FWHMG': FWHMG,
            'FWHML': FWHML,
            'Centroid': Centroid,
            'Background': Background,
            'Scale': Scale,
        }
        return returndict


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Spectrum simulator')
        self.mainTabs = QtWidgets.QTabWidget()

        self.data = np.loadtxt('testdata.txt', delimiter=',')
        self.setupFitterWidget()
        self.mainTabs.addTab(self.fitterWidget, 'Basic Fitter')

        self.setupSimulatorWidget()
        self.mainTabs.addTab(self.simulatorWidget, 'Simulator')
        self.setCentralWidget(self.mainTabs)

    def setupFitterWidget(self):
        self.fitterWidget = QtWidgets.QWidget()
        size = 1.5
        width, height = 16 / 2 * size, 9 / 2 * size
        self.fitterplot = FitterPlotter(self, width=width, height=height, dpi=150)
        toolbar = NavigationToolbar(self.fitterplot, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.fitterplot)
        layout.addWidget(toolbar)

        self.plotWidget = QtWidgets.QWidget()
        self.plotWidget.setLayout(layout)

        self.rightWidget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        self.paramWidget = ParameterWidget()
        self.pressButton = QtWidgets.QPushButton('Simulate')
        layout.addWidget(self.paramWidget)
        verticalSpacer = QtWidgets.QSpacerItem(20, 40,
                                               QtWidgets.QSizePolicy.Minimum,
                                               QtWidgets.QSizePolicy.Expanding)
        layout.addItem(verticalSpacer)
        layout.addWidget(self.pressButton)
        self.rightWidget.setLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.plotWidget)
        layout.addWidget(self.rightWidget)
        self.fitterWidget.setLayout(layout)
        self.pressButton.clicked.connect(self.updateData)
        self.paramWidget.sigChanged.connect(self.drawSingle)
        self.drawSingle()
    
    def updateFitter(self):
        self.drawFitter(False)

    def fitFitter(self):
        self.drawFitter(True)

    def drawFitter(self, fit):
        self.initSpectrumFitter()
        distance = self.fitParamWidget.getThreshold()
        data = self.data
        x = data[:, 0]
        y = data[:, 1]
        if self.fithfs is not None:
            try:
                splits = np.argwhere(np.diff(x) > distance)[0]
                right = x[splits+1]
                left = x[splits]
                left = np.append(x.min(), left)
                right = np.append(right, x.max())
            except:
                pass

            pos = self.hfs.pos()
            sorted_pos = np.sort(pos)
            fwhm, _ = self.hfs.calculateFWHM()
            left = sorted_pos - 3 * fwhm
            right = sorted_pos + 3 * fwhm
            remove = True
            while remove:
                innerLeft = left[1:]
                innerRight = right[:-1]
                try:
                    removePoints = np.argwhere(innerLeft < innerRight)[0]
                    left = np.delete(left, removePoints + 1)
                    right = np.delete(right, removePoints)
                except IndexError:
                    remove = False
        else:
            left = [0]
            right = [0]

        X = []
        for l, r in zip(left, right):
            X.append(np.linspace(l, r, 2))
        x = np.hstack(X)
        boundaries = []
        if len(left) > 1:
            for l, r in zip(left[1:], right[:-1]):
                boundaries.append((l + r) / 2)

        y = np.ones(x.shape)
        bunches = np.ones(y.shape)
        source = satlas2.Source(x,
                                y,
                                yerr=modifiedSqrt(y, bunches),
                                name='Data')
        f = CustomLlhFitter()
        f.addSource(source)
        if self.hfs is not None:
            source.addModel(self.hfs)
        source.addModel(self.background)
        self.f = f

        self.simulatorplot.drawSingle(f, boundaries)
        # self.paramWidget.updateLabels(f)
        self.simulatorplot.draw()

    def initSpectrum(self):
        returndict = self.fitterParamWidget.getParameters()
        spin = returndict['I']
        J = returndict['J']
        A = returndict['A']
        B = returndict['B']
        C = returndict['C']
        FWHMG = returndict['FWHMG']
        FWHML = returndict['FWHML']
        centroid = returndict['Centroid']
        bkg = returndict['Background']
        scale = returndict['Scale']

        try:
            self.fithfs = satlas2.HFS(spin,
                                   J,
                                   A,
                                   B,
                                   C,
                                   df=centroid,
                                   fwhmg=FWHMG,
                                   fwhml=FWHML,
                                   scale=scale)
        except:
            self.fithfs = None
        self.fitbackground = satlas2.Polynomial([bkg])

    def setupSimulatorWidget(self):
        self.simulatorWidget = QtWidgets.QWidget()
        size = 1.5
        width, height = 16 / 2 * size, 9 / 2 * size
        self.simulatorplot = SimulatorPlotter(self,
                                              width=width,
                                              height=height,
                                              dpi=150)
        toolbar = NavigationToolbar(self.simulatorplot, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.simulatorplot)
        layout.addWidget(toolbar)

        self.plotWidget = QtWidgets.QWidget()
        self.plotWidget.setLayout(layout)

        self.rightWidget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        self.paramWidget = ParameterWidget()
        self.pressButton = QtWidgets.QPushButton('Simulate')
        layout.addWidget(self.paramWidget)
        verticalSpacer = QtWidgets.QSpacerItem(20, 40,
                                               QtWidgets.QSizePolicy.Minimum,
                                               QtWidgets.QSizePolicy.Expanding)
        layout.addItem(verticalSpacer)
        layout.addWidget(self.pressButton)
        self.rightWidget.setLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.plotWidget)
        layout.addWidget(self.rightWidget)
        self.simulatorWidget.setLayout(layout)
        self.pressButton.clicked.connect(self.updateData)
        self.paramWidget.sigChanged.connect(self.drawSingle)
        self.drawSingle()

    @Qt.pyqtSlot()
    def updateData(self):
        samples, noise, step = self.paramWidget.getSamples()
        self.initSampling(samples, noise)
        self.initSpectrum()
        self.sampleSpectrum(step)

    @Qt.pyqtSlot()
    def drawSingle(self):
        self.initSpectrum()

        if self.hfs is not None:
            pos = self.hfs.pos()
            sorted_pos = np.sort(pos)
            fwhm, _ = self.hfs.calculateFWHM()
            left = sorted_pos - 3 * fwhm
            right = sorted_pos + 3 * fwhm
            remove = True
            while remove:
                innerLeft = left[1:]
                innerRight = right[:-1]
                try:
                    removePoints = np.argwhere(innerLeft < innerRight)[0]
                    left = np.delete(left, removePoints + 1)
                    right = np.delete(right, removePoints)
                except IndexError:
                    remove = False
        else:
            left = [0]
            right = [0]

        X = []
        for l, r in zip(left, right):
            X.append(np.linspace(l, r, 2))
        x = np.hstack(X)
        boundaries = []
        if len(left) > 1:
            for l, r in zip(left[1:], right[:-1]):
                boundaries.append((l + r) / 2)

        y = np.ones(x.shape)
        bunches = np.ones(y.shape)
        source = satlas2.Source(x,
                                y,
                                yerr=modifiedSqrt(y, bunches),
                                name='Data')
        f = CustomLlhFitter()
        f.addSource(source)
        if self.hfs is not None:
            source.addModel(self.hfs)
        source.addModel(self.background)
        self.f = f

        self.simulatorplot.drawSingle(f, boundaries)
        # self.paramWidget.updateLabels(f)
        self.simulatorplot.draw()

    def initSampling(self, samples, noise):
        bunches_background = samples
        self.sampling_noise = noise
        self.sampling_background = satlas2.Polynomial([bunches_background])

    def initSpectrum(self):
        returndict = self.paramWidget.getParameters()
        spin = returndict['I']
        J = returndict['J']
        A = returndict['A']
        B = returndict['B']
        C = returndict['C']
        FWHMG = returndict['FWHMG']
        FWHML = returndict['FWHML']
        centroid = returndict['Centroid']
        bkg = returndict['Background']
        scale = returndict['Scale']

        try:
            self.hfs = satlas2.HFS(spin,
                                   J,
                                   A,
                                   B,
                                   C,
                                   df=centroid,
                                   fwhmg=FWHMG,
                                   fwhml=FWHML,
                                   scale=scale)
        except:
            self.hfs = None
        self.background = satlas2.Polynomial([bkg])

    def sampleSpectrum(self, step):

        if self.hfs is not None:
            pos = self.hfs.pos()
            sorted_pos = np.sort(pos)
            models = [self.hfs, self.background]
            fwhm, _ = self.hfs.calculateFWHM()
            left = sorted_pos - 3 * fwhm
            right = sorted_pos + 3 * fwhm
            remove = True
            while remove:
                innerLeft = left[1:]
                innerRight = right[:-1]
                try:
                    removePoints = np.argwhere(innerLeft < innerRight)[0]
                    left = np.delete(left, removePoints + 1)
                    right = np.delete(right, removePoints)
                except IndexError:
                    remove = False
        else:
            models = [self.background]
            left = [0]
            right = [0]

        X = []
        for l, r in zip(left, right):
            X.append(np.arange(l, r + step, step))
        x = np.hstack(X)
        boundaries = []
        if len(left) > 1:
            for l, r in zip(left[1:], right[:-1]):
                boundaries.append((l + r) / 2)
        y = []
        rng = np.random.default_rng()
        mode = self.paramWidget.getSampleMode()
        if mode == 'constant':
            bunches = np.ones(x.shape) * self.sampling_background.f(0)
        else:
            if mode == 'gaussian':
                func = lambda x: rng.normal(x, self.sampling_noise)
            else:
                func = rng.poisson
            bunches = satlas2.generateSpectrum(self.sampling_background, x,
                                               func)
        bunches = np.abs(bunches)
        bunches = bunches.astype(int)
        bunches[bunches == 0] = 1
        for X, evaluated in zip(x, bunches):
            Y = satlas2.generateSpectrum(models, np.array([X] * evaluated),
                                         rng.poisson)
            y.append(Y.sum() / evaluated)
        y = np.array(y)

        if mode == 'constant':
            source = satlas2.Source(x,
                                    y,
                                    yerr=modifiedSqrt(y, bunches),
                                    name='Data',
                                    bunches_noplot=bunches)
        else:
            source = satlas2.Source(x,
                                    y,
                                    yerr=modifiedSqrt(y, bunches),
                                    name='Data',
                                    bunches=bunches)
        f = CustomLlhFitter()
        f.addSource(source)
        if self.hfs is not None:
            source.addModel(self.hfs)
        source.addModel(self.background)
        self.f = f

        self.simulatorplot.drawData(f, boundaries)
        self.paramWidget.updateLabels(f)
        self.simulatorplot.draw()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())