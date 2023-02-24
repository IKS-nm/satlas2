import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '..\src')

import satlas2
import satlas as sat

def modifiedSqrt(input):
    output = np.sqrt(input)
    output[input <= 0] = 1e-3
    return output

# single non-overlapping HF model

spin = 3.5
J = [0.5, 1.5]
A = [9600, 175]
B = [0, 315]
C = [0, 0]
FWHMG = 135
FWHML = 101
centroid = 480
bkg = [0.001, 100]
scale = 90

x = np.arange(-17500, -14500, 40)
x = np.hstack([x, np.arange(20000, 23000, 40)])

rng = np.random.default_rng(0)
hfs = satlas2.HFSModel(I = spin,
                  J = J,
                  ABC=[A[0], A[1], B[0], B[1], C[0], C[1]],
                  centroid = centroid,
                  fwhm = [FWHMG,FWHML],
                  scale = scale,
                  background_params = bkg,
                  use_racah=True
                  )
y = hfs(x)
y = rng.poisson(y)
hfs.params['centroid'].value = centroid - 100
hfs1 = sat.HFSModel(spin,
                    J,
                    [A[0], A[1], B[0], B[1], C[0], C[1]],
                    centroid - 100, [FWHMG, FWHML],
                    scale=scale,
                    background_params=bkg,
                    use_racah=True)
hfs1.set_variation({'Cu': False, 'Cl': False})

print('Fitting 1 dataset with chisquare (Pearson, satlas2)...')
start = time.time()
f = satlas2.chisquare_fit(hfs, x, y, modifiedSqrt(y), show_correl = False)
stop = time.time()
dt1 = stop - start

print('Fitting 1 dataset with chisquare (Pearson, satlas1)...')
start = time.time()
sat.chisquare_fit(hfs1, x, y, modifiedSqrt(y))
hfs1.display_chisquare_fit(show_correl = False)
stop = time.time()
dt2 = stop - start
print('SATLAS2: {:.3} s, {:.0f} function evaluations'.format(dt1, f.result.nfev))
print('SATLAS1: {:.3} s'.format(dt2))

fig, ((ax11,ax12),(ax21,ax22)) = plt.subplots(nrows = 2, ncols = 2, figsize = (14,9), sharex = True, sharey = True)
ax11.errorbar(x,y,modifiedSqrt(y), fmt = '.', label = 'Artificial data')
ax11.plot(x, hfs(x), '-', label = 'SATLAS2 fit')
ax11.set_xlim(-17500, -14500)
ax12.errorbar(x,y,modifiedSqrt(y), fmt = '.', label = 'Artificial data')
ax12.plot(x, hfs(x), '-', label = 'SATLAS2 fit')
ax12.set_xlim(20000, 23000)
ax21.errorbar(x,y,modifiedSqrt(y), fmt = '.', label = 'Artificial data')
ax21.plot(x, hfs1(x), '-', label = 'SATLAS1 fit')
ax21.set_xlim(-17500, -14500)
ax22.errorbar(x,y,modifiedSqrt(y), fmt = '.', label = 'Artificial data')
ax22.plot(x, hfs1(x), '-', label = 'SATLAS1 fit')
ax22.set_xlim(20000, 23000)
ax11.legend()
ax12.legend()
ax21.legend()
ax22.legend()
plt.show()

# Two overlapping HF models
J = [0.5, 1.5]
FWHMG = 135
FWHML = 101

spin1 = 4
A1 = [5300,100]
B1 = [0, 230]
C1 = [0, 0]
centroid1 = 400
bkg1 = 60
scale1 = 90

spin2 = 7
A2 = [3300,60]
B2 = [0, 270]
C2 = [0, 0]
centroid2 = -100
bkg2 = 30
scale2 = 160

x = np.arange(-13000, -9000, 40)
x = np.hstack([x, np.arange(11000, 14000, 40)])
rng = np.random.default_rng(0)

hfs1 = satlas2.HFSModel(I = spin1,
                  J = J,
                  ABC=[A1[0], A1[1], B1[0], B1[1], C1[0], C1[1]],
                  centroid = centroid1,
                  fwhm = [FWHMG,FWHML],
                  scale = scale1,
                  background_params = [bkg1],
                  use_racah=True
                  )
hfs2 = satlas2.HFSModel(I = spin2,
                  J = J,
                  ABC=[A2[0], A2[1], B2[0], B2[1], C2[0], C2[1]],
                  centroid = centroid2,
                  fwhm = [FWHMG,FWHML],
                  scale = scale2,
                  background_params = [bkg2],
                  use_racah=True
                  )
y = hfs1.f(x) + hfs2.f(x) + satlas2.Step([bkg1,bkg2],[0]).f(x)
y = rng.poisson(y)

hfs1.params['centroid'].value = centroid1 - 100
hfs2.params['centroid'].value = centroid2 - 100
summodel = satlas2.SumModel([hfs1,hfs2], {'values':[bkg1,bkg2], 'bounds':[0]})

print('Fitting 1 dataset with chisquare (Pearson, satlas2)...')
start = time.time()
f = satlas2.chisquare_fit(summodel, x, y, modifiedSqrt(y), show_correl = False)
stop = time.time()
dt1 = stop - start
print('SATLAS2: {:.3} s, {:.0f} function evaluations'.format(dt1, f.result.nfev))

fig, (ax1,ax2) = plt.subplots(ncols = 2, figsize = (14,9), sharey = True)
ax1.errorbar(x,y,modifiedSqrt(y), fmt = '.', label = 'Artificial data')
ax1.plot(x, hfs1.f(x), '-', label = 'SATLAS2 fit model 1')
ax1.plot(x, hfs2.f(x), '-', label = 'SATLAS2 fit model 2')
ax1.plot(x, summodel.f(x), '-', label = 'Sum of models')
ax1.set_xlim(-13000, -9000)
ax2.errorbar(x,y,modifiedSqrt(y), fmt = '.', label = 'Artificial data')
ax2.plot(x, hfs1.f(x), '-', label = 'SATLAS2 fit model 1')
ax2.plot(x, hfs2.f(x), '-', label = 'SATLAS2 fit model 2')
ax2.plot(x, summodel.f(x), '-', label = 'Sum of models')
ax2.set_xlim(11000, 14000)
ax1.legend()
ax2.legend()
plt.show()