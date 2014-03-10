import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1,2)

x = np.linspace(0, 1, 3)
y = np.linspace(0, 2, 2)
X, Y = np.meshgrid(x, y)
Z = X * Y



levels = np.linspace(-1, 1, 40)

zdata = np.sin(8*X)*np.sin(8*Y)

cs = axs[0].contourf(X, Y, zdata, levels=levels)
fig.colorbar(cs, ax=axs[0], format="%.2f")

cs = axs[1].contourf(X, Y, zdata, levels=[-1,0,1])
fig.colorbar(cs, ax=axs[1])

plt.show()
