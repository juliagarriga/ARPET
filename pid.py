import time

import cv2
import numpy as np


class PID:
    
    def __init__(self, vmax, w, h, kp, ki, kd):
        self.vmax = vmax
        self.w = w
        self.h = h
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.sum = 0
    
    def speed(self, e, prev_e):
        self.sum += e
        return 0.5 * self.vmax * (self.kp * (e / (self.w/2)) + self.kd * ((e - prev_e) / self.w))
    
    def speed2(self, e, prev_e):
        self.sum += e
        return self.vmax * np.tanh(self.kp*e + self.ki*self.sum + self.kd*(e - prev_e))

    def increase_kp(self, increase):
        self.kp += increae

    def increase_kd(self, increase):
        self.kd += increase

    def increase_ki(self, increase):
        self.ki += increase