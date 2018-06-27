import numpy as np
import matplotlib.pyplot as plt

with open("textfiles/file_tracker1.032364.txt") as f:
	tracker = f.read().split('\n')[:-1]


with open("textfiles/file_kalman1.032364.txt") as f:
	kalman = f.read().split('\n')[:-1]

with open("textfiles/file_time1.032364.txt") as f:
	time = f.read().split('\n')[:-1]

tracker = [eval(x) for x in tracker]
kalman = [eval(x) for x in kalman]
time = np.array([eval(x) for x in time])

x_t, y_t = np.array(tracker).T
x_k, y_k = np.array(kalman).T

fig = plt.figure()

ax1 = fig.add_subplot(111)   
ax1.set_xlabel('time (s)')
ax1.set_ylabel('x-coordinate (pixels)')

step = 86./len(x_t)
x = np.arange(0.0, 86.0, step)
print min(x_t), max(x_t)
print min(x_k), max(x_k)
print min(y_t), max(y_t)
print min(y_k), max(y_k)

ax1.plot(x, x_t, 'r', label='Tracker data', )
ax1.plot(x, x_k, 'b', label='Kalman data')

leg = ax1.legend()
plt.show()