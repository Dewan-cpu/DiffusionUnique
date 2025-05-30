# examples/run_diffusion.py

import numpy as np
import matplotlib.pyplot as plt
from diffusion2d import DiffusionModel2D

x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
field = 5.0 * np.exp(-4 * (X**2 + Y**2))

model = DiffusionModel2D(field, spacing=(1.0, 1.0), diffusivity=0.1)

dt = model.calc_stable_dt()
for _ in range(int(3600 / dt)):
    model.step(dt)

plt.imshow(model.field, origin='lower', extent=[-2, 2, -2, 2])
plt.title("Final Field")
plt.colorbar()
plt.show()

        
        
