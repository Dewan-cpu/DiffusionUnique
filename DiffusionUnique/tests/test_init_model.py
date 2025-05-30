# diffusion2d/model.py

import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys
import DiffusionUnique as diff

class TestDiffusionModel2DInit:
    
    # def test_init_t0(self):
    #    x = np.linspace(-2, 2, 20)
    #    y = np.linspace(-2, 2, 20)
    #    X, Y = np.meshgrid(x, y)
    #    field = 5.0 * np.exp(-4 * (X**2 + Y**2))

    #    model = diff.DiffusionModel2D(field, spacing=(1.0, 1.0), diffusivity=0.1)

    #    Think

       def test_init_t0(self):
            x = np.linspace(-2, 2, 20)
            y = np.linspace(-2, 2, 20)
            X, Y = np.meshgrid(x, y)
            field = 5.0 * np.exp(-4 * (X**2 + Y**2))
            model = diff.DiffusionModel2D(field, spacing=(1.0, 1.0), diffusivity=0.1)
            #mod = DiffusionModel2D(field, spacing=(1.0, 1.0), diffusivity=0.1)
            ny, nx = field.shape
            assert model.nx == nx
            assert model.ny == ny

        

   
