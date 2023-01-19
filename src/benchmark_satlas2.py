import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''Make sure the directory to the folder with the data is path_to_data\\mass\\'''

path_to_data = f'C:\\Users\\u0148746\\OneDrive - KU Leuven\\Documents\\PhD\\2022-2023\\Miscellaneous tasks\\Benchmark satlas2\\'
spectra = {
    110: {
        1: [['4811', '4814'], ['4799', '4806']],
        6: [['4798', '4804'], ['4812', '4813']]
    },
    108: {
        6: [['4817', '4819']]
    }
}
spectra = {111: {3.5: [['4648', '4649'], ['4688', '4689'], ['4845', '4846']]}}
HF_constants = {
    110: {
        1: [21100, 360, 0, 40, 0, 0],
        6: [4660, 80, 0, 300, 0, 0]
    },
    108: {
        6: [4630, 80, 0, 400, 0, 0]
    },
    111: {
        3.5: [9675, 180, 0, 300, 0, 0]
    }
}  #[A_lower,A_upper,B_lower,B_upper,C_lower,C_upper]
J = [0.5, 1.5]

mass = 111
spin = 3.5


def extract_data(path, mass, lscans):
    if len(lscans) == 1:
        scan = lscans[0]
        data = pd.read_csv(f'{path_to_data}{mass}\\{scan}.csv',
                           delimiter=';',
                           header=0,
                           names=['x', 'y', 'xerr', 'yerr'])
        return data['x'], data['y'], data['xerr'], data['yerr']
    else:
        scan_L = lscans[0]
        scan_R = lscans[1]
        data_L = pd.read_csv(f'{path_to_data}{mass}\\{scan_L}.csv',
                             delimiter=';',
                             header=0,
                             names=['x', 'y', 'xerr', 'yerr'])
        data_R = pd.read_csv(f'{path_to_data}{mass}\\{scan_R}.csv',
                             delimiter=';',
                             header=0,
                             names=['x', 'y', 'xerr', 'yerr'])
        data = np.concatenate((data_L, data_R)).T
        return data[0], data[1], data[2], data[3]


x, y, xerr, yerr = extract_data(path_to_data, mass, spectra[mass][spin][0])
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(14, 9), sharey=True)
ax0.errorbar(x,
             y,
             xerr=xerr,
             yerr=yerr,
             fmt='r.',
             capsize=2,
             ecolor='k',
             markersize=10,
             fillstyle='none',
             label=f'I = {spin}')
ax1.errorbar(x,
             y,
             xerr=xerr,
             yerr=yerr,
             fmt='r.',
             capsize=2,
             ecolor='k',
             markersize=10,
             fillstyle='none',
             label=f'I = {spin}')
plt.legend(fontsize=16)
plt.suptitle(r'$^{' + str(mass) + '}$Ag', fontsize=20)
ax0.set_ylabel('Counts per ion bunch', fontsize=20)
ax0.set_xlabel('Relative frequency [MHz]', fontsize=20)
ax0.set_xlim(-19000, -14000)
ax1.set_xlim(20000, 23000)
plt.show()
