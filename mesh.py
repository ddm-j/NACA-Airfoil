import numpy as np
import matplotlib.pyplot as plt
from naca4digit import NACA4


# Create an Airfoil
af = NACA4('4412')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(af.ordered_points[:, 0], af.ordered_points[:, 1], af.ordered_points[:, 2])
plt.show()