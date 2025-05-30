# diffusion2d/model.py

import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys
import DiffusionUnique as diff

class TestDiffusionModel2DStep:
    
    def test_step_runs_without_error(self):
        # Create initial Gaussian field
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)
        field = 5.0 * np.exp(-4 * (X**2 + Y**2))

        # Initialize model
        model = diff.DiffusionModel2D(field, spacing=(1.0, 1.0), diffusivity=0.1)

        # Calculate stable dt
        dt = model.calc_stable_dt()

        # Run only a few steps for testing
        for _ in range(10):
            model.step(dt)

        # Check if field was updated (not same as initial)
        assert not np.allclose(model.field, field), "Field did not change after stepping."

    

    
