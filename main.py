import argparse
import time

import cv2
import numpy as np
import ps_drone

import tracker
import keyboard


# Initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
  "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
  "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
  "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def init_drone():
    
    # Connect
    drone = ps_drone.Drone()
    drone.startup()
    
    # Reset
    drone.reset()

    while drone.getBattery()[0] == -1:
        time.sleep(0.1)
        
    time.sleep(0.5)
    
    # Calibrate
    drone.trim()
    drone.getSelfRotation(5)
    
    # Configure video
    cdc0 = drone.ConfigDataCount
    
    drone.setConfigAllID()
    drone.hdVideo()
    drone.frontCam()

    while drone.ConfigDataCount == cdc0:
        time.sleep(0.1)
        
    drone.startVideo()
    
    return drone, (drone.getBattery()[1] != 'empty')


def frame_grabber(drone):

    imc = drone.VideoImageCount

    while True:
        
        if drone.VideoImageCount == imc:
            yield None, None
            
        else:
            imc = drone.VideoImageCount
            frame = cv2.cvtColor(drone.VideoImage, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            t = drone.VideoDecodeTimeStamp + drone.VideoDecodeTime
            yield frame, t

def main(): 

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # Load our serialized model from disk
    print("[INFO] Loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # Initialize the drone and allow the camera sensor to warm up
    drone, status = init_drone()

    if not status:
        print "[INFO] No battery"
        #return None
    frames = frame_grabber(drone)
    time.sleep(2.0)

    drone.takeoff()

    time.sleep(5.0)

    vmax = 0.4
    w, h = 180, 320 
    #kpx, kdx = 0.0025, 0.05

    kpx, kdx, kix = 0., 0., 0.
    kpx, kdx, kix = 0.002, 0.04, 0.
    kpy, kdy, kiy = 0.004, 0., 0.
    kpz, kdz, kiz = 0., 0., 0.
    kpz, kdz, kiz = 0.005, 0.05, 0.00001
    kpt, kdt, kit = 0., 0., 0.

    kp = [kpx, kpy, kpz, kpt]
    kd = [kdx, kdy, kdz, kdt]
    ki = [kix, kiy, kiz, kit]

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('videos/tracking' + str(time.clock()) + '.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 15., (3*h, 3*w))
    track_person = tracker.Tracker("Tracking", drone, net, frames, vmax, w, h, kp, ki, kd, args['confidence'], 80)

    track_person.track(out)

    drone.land()
    out.release()

main()