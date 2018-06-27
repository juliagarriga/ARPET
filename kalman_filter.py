import time

import numpy as np


class KalmanFilterPixel:
    
    def __init__(self, ax, ay, rx, ry, px0, py0, pu0, pv0):
        """
        Linear Kalman filter for single pixel tracking. It assumes a constant velocity model.
        The state vector is (x, y, u, v) where (x, y) is the pixel position and (u, v) is the
        pixel velocity (in pixels per second). The acceleration of the pixel is taken to have
        a diagonal (uncorrelated x and y terms) covariance. The observation vector is assumed
        to contain only the pixel position, and its error to have also a diagonal covariance.
        The initial velocity is assumed to be zero.
        
        ax: Std. deviation of the pixel x-acceleration.
        ay: Std. deviation of the pixel y-acceleration.
        rx: Std. deviation of the pixel x-position observation.
        ry: Std. deviation of the pixel y-position observation.
        px0: Std. deviation of the pixel initial x-position.
        py0: Std. deviation of the pixel initial y-position.
        pu0: Std. deviation of the pixel initial x-velocity.
        pv0: Std. deviation of the pixel initial y-velocity.
        
        """
        self.ax = ax
        self.ay = ay
        
        self.rx = rx
        self.ry = ry
        
        # Initial state covariance matrix
        self.P = np.array([[px0*px0, 0, 0, 0],
                           [0, py0*py0, 0, 0],
                           [0, 0, pu0*pu0, 0],
                           [0, 0, 0, pv0*pv0]])
        
        # State vector
        self.x = None
        
        # Linear model matrices
        self.F = None
        self.H = None
        self.Q = None
        self.R = None
        
        # Time of the last observation
        self.t0 = None
            
    def filter_pixel(self, z, t, sleep_time=None):
        """
        Fuses the given measurement data with the linear model prediction to estimate
        the current pixel state.
        
        z: Observed pixel position (x, y).
        t: Time of the observation.
        sleep_time: (Optional) Time to sleep before the function returns, in seconds.
        
        """
        if self.t0 is None:
            
            self.t0 = t
            self.x = np.array(list(z) + [0, 0])
            
            return self.x
        
        self._compute_matrices(t - self.t0)
        self._predict()
        self._update(z)
        
        self.t0 = t
        
        if sleep_time is not None:
            time.sleep(sleep_time)
        
        return self.x
        
    def _compute_matrices(self, delta_t):

        delta_t2 = delta_t * delta_t
        delta_t3 = delta_t * delta_t2 / 2
        delta_t4 = delta_t2 * delta_t2 / 4

        ax = self.ax * self.ax
        ay = self.ay * self.ay

        rx = self.rx * self.rx
        ry = self.ry * self.ry

        # State transition matrix
        F = np.array([[1, 0, delta_t, 0],
                      [0, 1, 0, delta_t],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        # Observation model matrix
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        # Process noise covariance matrix
        Q = np.array([[delta_t4*ax, 0, delta_t3*ax, 0],
                      [0, delta_t4*ay, 0, delta_t3*ay],
                      [delta_t3*ax, 0, delta_t*ax, 0],
                      [0, delta_t3*ay, 0, delta_t*ay]])

        # Observation noise covariance matrix
        R = np.array([[rx, 0],
                      [0, ry]])
        
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        
    def _predict(self):
        
        self.x = np.matmul(self.F, self.x)
        self.P = np.matmul(np.matmul(self.F, self.P), np.transpose(self.F)) + self.Q
        
    def _update(self, z):
        
        y = z - np.matmul(self.H, self.x)
        S = np.matmul(np.matmul(self.H, self.P), np.transpose(self.H)) + self.R
        K = np.matmul(np.matmul(self.P, np.transpose(self.H)), np.linalg.inv(S))
        
        self.x = self.x + np.matmul(K, y)
        
        tmp = np.eye(4) - np.matmul(K, self.H)
        self.P = np.matmul(np.matmul(tmp, self.P), np.transpose(tmp))
        self.P = self.P + np.matmul(np.matmul(K, self.R), np.transpose(K))