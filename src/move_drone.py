import ps_drone

import pid
import keyboard


class MoveDrone:

	def __init__(self, drone, pidx, pidy, pidz, pidt):
		self.drone = drone
		self.pidx = pidx
		self.pidy = pidy
		self.pidz = pidz
		self.pidt = pidt
		self.vx = 0.
		self.vy = 0.
		self.vz = 0.
		self.vt = 0.

	def roll(self, e, prev_e):
		self.vx = self.pidx.speed2(e, prev_e)

	def gaz(self, e, prev_e):
		self.vy = self.pidy.speed2(e, prev_e)

	def pitch(self, e, prev_e):
		self.vz = self.pidz.speed2(e, prev_e)

	def yaw(self, e, prev_e):
		self.vt = self.pidt.speed2(e, prev_e)          

	def set_velocities(self, ex, prev_ex, ey, prev_ey, ez, prev_ez, et, prev_et):
		self.roll(ex, prev_ex)
		self.gaz(ey, prev_ey)
		self.pitch(ez, prev_ez)
		self.yaw(et, prev_et)

	def move_drone(self):
		print "VX: ", self.vx
		print "VY: ", self.vy
		print "VZ: ", self.vz
		self.drone.move(float(self.vx), float(self.vz), float(self.vy), float(self.vt))

	def stop_drone(self):
		self.drone.stop()