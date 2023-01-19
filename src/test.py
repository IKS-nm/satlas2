import satlas2 as sat
import matplotlib.pyplot as plt
import numpy as np

def modifiedSqrt(input):
    output = np.sqrt(input)
    output[input<0] = 1e-12
    return output

J = [0.5, 1.5]
I = 0

fwhm = 30
powers = [0.5, 0.76, 1.0, 1.5, 2.0, 2.5]

# Basic: fit each separate spectrum, and make a plot of the scale as a function of laser power
f = sat.Fitter()
for power in powers:
    data = np.loadtxt('test-data/{}mW_power_1810.txt'.format(power))
    x, y = data[:, 0], data[:, 1]
    datasource = sat.Source(x, y, yerr=modifiedSqrt, name='P{:d}uW'.format(int(1000*power)))
    Yb174_model = sat.HFS(I, J, name='Yb174', df=x.mean(), scale=y.ptp())
    datasource.addModel(Yb174_model)
    f.addSource(datasource)
# f.shareParams(['FWHMG', 'FWHML', 'bkg'])
f.prepareFit()
m = f.createMinuit()
f.temp_y = f.y()
import time
s = time.time()
# m.migrad()
f.fit()
st = time.time()
# print(f.reportFit())
print(st-s)

# fig = plt.figure()
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# for datasource in f.sources:
#     datasource = datasource[1]
#     ax.plot(datasource.x, datasource.y, label='Data', drawstyle='steps-mid')
#     ax.plot(datasource.x, datasource.evaluate(datasource.x), label='Optim')
# plt.show()