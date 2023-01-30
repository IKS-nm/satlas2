import numpy as np
import matplotlib.pyplot as plt
import satlas2
import time
import satlas as sat
import emcee
'''Make sure the directory to the folder with the data is path_to_data\\mass\\'''

path_to_data = ''
spectra = {
    110: {
        1: [['4811', '4814'], ['4799', '4806']],
        6: [['4798', '4804'], ['4812', '4813']]
    },
    108: {
        6: [['4817', '4819']]
    },
    111: {
        3.5: [['4648', '4649'], ['4688', '4689'], ['4845', '4846']]
    }
}
HF_constants = {
    110: {
        1: [21100, 360, 0, 40, 0, 0],
        6: [4660, 80, 0, 300, 0, 0]
    },
    108: {
        6: [4630, 80, 0, 400, 0, 0]
    },
    111: {
        3.5: [9676, 175, 0, 313, 0, 0]
    }
}  #[A_lower, A_upper, B_lower, B_upper, C_lower, C_upper]
J = [0.5, 1.5]

mass = 111
spin = 3.5


def extract_data(path, mass, lscans):
    if len(lscans) == 1:
        scan = lscans[0]
        data = np.loadtxt(f'{path}{mass}\\{scan}.csv',
                          delimiter=';',
                          skiprows=1)
        x = data[:, 1]
        y = data[:, 2]
        xerr = data[:, 3]
        yerr = data[:, 4]
        return x, y, xerr, yerr
    else:
        scan_L = lscans[0]
        scan_R = lscans[1]
        dataL = np.loadtxt(f'{path}{mass}\\{scan_L}.csv',
                           delimiter=';',
                           skiprows=1)
        dataR = np.loadtxt(f'{path}{mass}\\{scan_R}.csv',
                           delimiter=';',
                           skiprows=1)
        data = np.vstack([dataL, dataR])
        x = data[:, 1]
        y = data[:, 2]
        xerr = data[:, 3]
        yerr = data[:, 4]
        return x, y, xerr, yerr


f = satlas2.Fitter()
i = 0
models = []
X = []
Y = []
YERR = []
for _ in range(2):
    for scancouple in spectra[mass][spin]:
        x, y, xerr, yerr = extract_data(path_to_data, mass, scancouple)
        X.append(x)
        Y.append(y)
        YERR.append(yerr)
        datasource = satlas2.Source(x,
                                    y,
                                    yerr=yerr,
                                    name='Scan{}{}'.format(scancouple[0], i))
        hfssat1 = sat.HFSModel(spin,
                               J,
                               HF_constants[mass][spin],
                               425,
                               scale=y.ptp(),
                               background_params=[y.min()])
        models.append(hfssat1)
        hfs = satlas2.HFS(
            spin,
            J,
            A=HF_constants[mass][spin][:2],
            B=HF_constants[mass][spin][2:4],
            C=HF_constants[mass][spin][4:],
            #   bkg=y.min(),
            scale=y.ptp(),
            df=425,
            name='Ag' + str(mass) + str(spin).replace('.', '_'),
            racah=False)
        datasource.addModel(hfs)
        bkg = satlas2.Polynomial([y.min()],
                                 name='Ag' + str(mass) +
                                 str(spin).replace('.', '_') + 'bkg')
        datasource.addModel(bkg)
        f.addSource(datasource)
        i += 1
model = sat.LinkedModel(models)
startsat = time.time()
# success, message = sat.chisquare_fit(model, X, Y, YERR)
# print(message)
# model.display_chisquare_fit(show_correl=False)
stopsat = time.time()
fig, axes = plt.subplots(ncols=2, figsize=(14, 9), sharey=False, nrows=i)
# f.shareModelParams(['Al', 'Au', 'Bl', 'Bu'])
# f.prepareFit()
# f.setExpr('Scan4648___Ag1113_5___Au', '0.0181664043527959*Scan4648___Ag1113_5___Al')
start = time.time()
f.fit()
stop = time.time()
# f.fit(llh_selected=True, method='slsqp')
print(f.reportFit())
print('Fitting time (satlas2): {:.3f}s'.format(stop - start))
print('Fitting time (satlas1): {:.3f}s'.format(stopsat - startsat))
# print(f.sources)
# print('Starting normal walk...')
# f.fit(llh_selected=True,
#       method='emcee',
#       llh_method='poisson',
#       filename='benchmark_normal.h5',
#       steps=500,
#       nwalkers=160,
#       )
# f.fit()
# print('Starting dangerous walk...')
# f.fit(llh_selected=True,
#       method='emcee',
#       llh_method='poisson',
#       filename='benchmark_dangerous.h5',
#       steps=1000,
#       nwalkers=80,
#       sampler_kwargs={'moves': [
#         (emcee.moves.StretchMove(live_dangerously=True), 1.0)
#       ]}
#       )
# f.fit()
print('Starting DE tuning walk...')
f.fit(llh_selected=True,
      method='emcee',
      llh_method='poisson',
      filename='benchmark_normal_tune.h5',
      steps=500,
      nwalkers=160,
    #   sampler_kwargs={
    #       'moves': [(emcee.moves.DEMove(), 0.8),
    #                 (emcee.moves.DESnookerMove(), 0.2)]
    #   },
      mcmc_kwargs={'tune': True},
      )
# f.fit()
# print('Starting dangerous DE walk...')
# f.fit(llh_selected=True,
#       method='emcee',
#       llh_method='poisson',
#       filename='benchmark_de_dangerous.h5',
#       steps=1000,
#       nwalkers=80,
#       sampler_kwargs={'moves': [
#         (emcee.moves.DEMove(live_dangerously=True), 0.8),
#         (emcee.moves.DESnookerMove(live_dangerously=True), 0.2)
#       ]}
#       )
# print(f.reportFit())

for i, (name, source) in enumerate(f.sources):
    ax0 = axes[i][0]
    ax1 = axes[i][1]
    x = source.x
    y = source.y
    yerr = source.yerr()
    ax0.errorbar(
        x,
        y,
        # xerr=xerr,
        yerr=yerr,
        fmt='r.',
        capsize=2,
        ecolor='k',
        markersize=10,
        fillstyle='none',
        label=f'I = {spin}')
    ax1.errorbar(
        x,
        y,
        # xerr=xerr,
        yerr=yerr,
        fmt='r.',
        capsize=2,
        ecolor='k',
        markersize=10,
        fillstyle='none',
        label=f'I = {spin}')
    # plt.legend(fontsize=16)
    # plt.suptitle(r'$^{' + str(mass) + '}$Ag', fontsize=20)
    if i == len(f.sources) - 1:
        ax0.set_ylabel('Counts per ion bunch', fontsize=20)
        ax0.set_xlabel('Relative frequency [MHz]', fontsize=20)
    ax0.plot(source.x, source.evaluate(source.x))
    ax1.plot(source.x, source.evaluate(source.x))
    ax0.set_xlim(-19000, -14000)
    ax1.set_xlim(19000, 24000)
# satlas2.generateCorrelationPlot('benchmark.h5', selection=(20, 100), source=True)
# satlas2.generateCorrelationPlot('benchmark.h5', burnin=100, filter=['Al'], source=False, binreduction=2, bin2dreduction=1)
satlas2.generateCorrelationPlot('benchmark_dangerous.h5',
                                burnin=100,
                                filter=['Al'],
                                source=False,
                                binreduction=2,
                                bin2dreduction=2)
# satlas2.generateCorrelationPlot('benchmark.h5', burnin=100, filter=['Al'], source=False, binreduction=2, bin2dreduction=3)
# satlas2.generateCorrelationPlot('benchmark.h5', burnin=100, filter=['Al'], source=False, binreduction=2, bin2dreduction=4)
satlas2.generateWalkPlot('benchmark_dangerous.h5',
                         filter=['centroid'],
                         burnin=0)
plt.show()
