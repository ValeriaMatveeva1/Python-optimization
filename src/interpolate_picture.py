import numpy as np
import matplotlib.pyplot as plt
from algorithms import interpolate_in_point


x = np.array([2, 5, -6, 7, 4, 3, 8, 9, 1, -2], dtype=float)
y = np.array([-1, 77, -297, 249, 33, 9, 389, 573, -3, -21], dtype=float)
xnew = np.linspace(min(x), max(x), 100)
ynew = [interpolate_in_point(x, y, i) for i in xnew]
plt.plot(x, y, 'o', xnew, ynew)
plt.grid(True)
plt.show()
