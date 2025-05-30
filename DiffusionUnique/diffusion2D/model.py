# diffusion2d/model.py

import numpy as np

class DiffusionModel2D:
    def __init__(self, field, spacing=(1.0, 1.0), diffusivity=0.1):
        '''
        

        Parameters
        ----------
        field : numpy.ndarray
            A 2D array represtening the initial state of the field.
        spacing : float, optional
            The default is (1.0, 1.0).
        diffusivity : float, optional
            The default is 0.1.

        Returns
        -------
        None.

        '''
        self.field = field.copy()
        self.dx, self.dy = spacing
        self.D = diffusivity
        self.ny, self.nx = field.shape

    def calc_stable_dt(self):
        '''
        

        Returns
        -------
        TYPE
            Calculate the maximum stable time step for the diffusion simulation based on the Courant–Friedrichs–Lewy (CFL) condition.

        '''
        dx2, dy2 = self.dx**2, self.dy**2
        return (dx2 * dy2) / (2 * self.D * (dx2 + dy2))

    def step(self, dt):
        '''
        

        Parameters
        ----------
        dt : float
            The time step duration. Should be less than or equal to the stable time step detrmined by CFL condition.

        Returns
        -------
        None.

        '''
        new_field = self.field.copy()
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                d2x = (self.field[i, j-1] - 2*self.field[i, j] + self.field[i, j+1]) / self.dx**2
                d2y = (self.field[i-1, j] - 2*self.field[i, j] + self.field[i+1, j]) / self.dy**2
                new_field[i, j] += self.D * dt * (d2x + d2y)
        self.field = new_field
