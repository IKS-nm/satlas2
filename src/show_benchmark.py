import satlas2
import matplotlib.pyplot as plt

def showcase(filename):
    if 'dangerous' in filename:
        burnin = 1000-2*80
    else:
        burnin = 500-2*160
    burnin = 250
    satlas2.generateCorrelationPlot(filename, burnin=burnin, filter=['Al'], source=False, binreduction=2, bin2dreduction=2)
    satlas2.generateWalkPlot(filename, filter=['Al'], burnin=0)

# showcase('benchmark_normal.h5')
# showcase('benchmark_dangerous.h5')
showcase('benchmark_de.h5')
showcase('benchmark_de_tune.h5')

plt.show()