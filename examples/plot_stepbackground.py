import sys

sys.path.insert(0, "..\src")
import time

import matplotlib.pyplot as plt
import numpy as np

import satlas2

model = satlas2.Step([0, 1, 2], [3, 5])

x = np.linspace(0, 10, 200)
start = time.time()
for _ in range(1000):
    y = model.f(x)
stop = time.time()

plt.plot(x, y)
plt.show()
