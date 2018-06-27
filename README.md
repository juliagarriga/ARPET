# ARPET

Implementation of an onboard vision system autnonomous tracker for an AR.Drone 2.0 using three PID controllers and the 
PS-Drone http://www.playsheep.de/drone/ library. The tracking algorithm is composed of an object detector implemented using 
a Caffe trained network of MobileNet-SSD, and a KCF object tracker using OpenCV v3. A Kalman filter is implemented to reduce
the noise from the tracker and the quadcopter.

All code is written in Python 2.7. The dependencies are the PS-Drone library, OpenCV version > 3.0 and Numpy.
