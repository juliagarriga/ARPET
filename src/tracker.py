import time

import cv2
import numpy as np

import kalman_filter
import move_drone
import pid


class Tracker:

    def __init__(self, window_name, drone, net, frames, vmax, w, h, kp, ki, kd, tol, alpha):
        
        self.window_name = window_name
        self.net = net
        self.tol = tol
        self.frames = frames
        self.alpha = alpha
        self.kalman = self.init_kalman_filter(4, 4)
        self.h = h

        t = str(time.clock())

        # Files used to create the Kalman filter graphics
        self.file_tracker = open("textfiles/file_tracker" + t + ".txt", "w")
        self.file_time = open("textfiles/file_time" + t + ".txt", "w")
        self.file_kalman = open("textfiles/file_kalman" + t + ".txt", "w")

        pidx = pid.PID(vmax, w, h, kp[0], ki[0], kd[0])
        pidy = pid.PID(vmax, w, h, kp[1], ki[1], kd[1])
        pidz = pid.PID(vmax, w, h, kp[2], ki[2], kd[2])
        pidt = pid.PID(vmax, w, h, kp[3], ki[3], kd[3])

        self.controller = move_drone.MoveDrone(drone, pidx, pidy, pidz, pidt)

    def init_kalman_filter(self, ax = 5, ay = 5, rx = 10, ry = 10, px0 = 20, py0 = 20, pu0 = 20, pv0 = 20):

        return kalman_filter.KalmanFilterPixel(ax, ay, rx, ry, px0, py0, pu0, pv0)

    def locate(self, frame):

        #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        # Grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame , 0.007843, (300, 300), 127.5)

        # Pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        found_persons = []

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
        
            # extract the index of the class label from the 'detections'
            idx = int(detections[0, 0, i, 1])

            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if idx == 15 and confidence > self.tol:
              # compute the (x, y)-coordinates of
              # the bounding box for the object
              box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
              found_persons.append(box)

        return found_persons

    def update_frame(self, frame, tl, br):

        croppedX = max(tl[0] - self.alpha, 0) 
        croppedY = max(tl[1] - self.alpha, 0)

        cropped_image = frame[croppedY: br[1] + self.alpha, croppedX:br[0] + self.alpha]

        found_persons = self.locate(cropped_image)

        if found_persons:

            (startX, startY, endX, endY) = found_persons[0].astype(int)

            rect = np.array([startX + croppedX, startY + croppedY, endX - startX, endY - startY])# Detection 

            return rect

        return None

    def track_object(self, frame, out, rect):

        # Create tracker
        tracker = cv2.TrackerKCF_create()

        tracking_height = min(rect[3], self.h*0.8)
        # Initialize tracker with frame and bounding box
        ok = tracker.init(frame, tuple(rect))

        failure = 0

        t0 = time.time()
        t1 = t0

        print "Tracking height: ", tracking_height
        height = tracking_height

        prev_ex, prev_ey, prev_ez = 0, 0, 0

        while True:

            # Read a new frame
            frame, t = self.frames.next()

            if frame is not None:
                # Start timer
                timer = cv2.getTickCount()

                # Update tracker
                ok, bbox = tracker.update(frame)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

                elapsed_bbox = time.time() - t0

                elapsed_frame = time.time() - t1

                if ok:
                    # Tracking success
                    tl = (int(bbox[0]), int(bbox[1]))
                    br = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                    # Every half second recalculate the person's bounding box
                    # using the person's computed bounding box enlarged a certain alpha
                    # to obtain a bigger bounding box if the person is moving forwards
                    # or a smaller one if the person is moving backwards

                    if elapsed_bbox > 0.5:

                        rect = self.update_frame(frame, tl, br)

                        if rect is not None:

                            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2] + rect[0], rect[3] + rect[1]), (0,255,0), 2, 1)
            
                            # Recreate tracker
                            tracker = cv2.TrackerKCF_create()

                            # Initialize tracker with frame and bounding box
                            ok = tracker.init(frame, tuple(rect))

                            height = rect[3]

                            print "HEIGHT: ", height

                        t0 = time.time()

                    #if elapsed_frame > 0.05:

                    px = tl[0] + int(br[0] - tl[0])/2
                    py = tl[1] + int(br[1] - tl[1])/2

                    self.file_tracker.write(str((px, py)) + '\n')

                    self.file_time.write(str(t) + '\n')
                    px_filtered, py_filtered = np.round(self.kalman.filter_pixel((px, py), t)[:2]).astype(np.int)
                    self.file_kalman.write(str((px_filtered, py_filtered)) + '\n')

                    tl_filtered = (tl[0] + px_filtered - px, tl[1] + py_filtered - py)
                    br_filtered = (br[0] + px_filtered - px, br[1] + py_filtered - py)
                    
                    # If the frame vertical center is smaller than the x coordinate of the
                    # observation, the error has to be positive for the drone to go right.
                    ex = px_filtered - frame.shape[1]/2
                    # If the frame horizontal center is bigger than the y coordinate of the 
                    # observation, the error has to be positive for the drone to go up.
                    ey = frame.shape[0]/1.8 - py_filtered 

                    self.controller.set_velocities(ex, prev_ex, ey, prev_ey, tracking_height - height, prev_ez, 0, 0)

                    self.controller.move_drone()

                    prev_ez = tracking_height - height
                    prev_ex = ex
                    prev_ey = ey
                    t1 = time.time()

                    # Tracking bounding box
                    cv2.rectangle(frame, tl, br, (255,0,0), 2, 1)
                    # Tracking filtered bounding box
                    cv2.rectangle(frame, tl_filtered, br_filtered, (0,0,255), 2, 1)

                else :      
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure detected", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255),2)

                    self.controller.stop_drone()
                    failure += 1
                    if failure > 50:
                        return False

                # Display tracker type on frame
                #cv2.putText(frame, "KCF Tracker", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50),2);

                # Display FPS on frame
                #cv2.putText(frame, "FPS : " + str(int(fps)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 2);

                # Display result
                frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=3, fy=3), cv2.COLOR_RGB2BGR)
                cv2.imshow(self.window_name, frame) 
                if out is not None:
                    out.write(frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break

        self.file_tracker.close()
        self.file_kalman.close()
        self.file_time.close()
        return True

    def track(self, out=None):

        cv2.namedWindow(self.window_name)

        stop = False

        while not stop:

            # Obtain new frame
            frame, t = self.frames.next()

            if frame is not None:

                found_persons = self.locate(frame)

                frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=3, fy=3), cv2.COLOR_RGB2BGR)
                cv2.imshow(self.window_name, frame)

                if out is not None:
                    out.write(frame) 

                if found_persons:
                    (startX, startY, endX, endY) = found_persons[0].astype(int)

                    # Detection bounding box
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2, 1)

                    rect = np.array([startX, startY, endX - startX, endY - startY])           

                    stop = self.track_object(frame, out, rect)

                # Exit if ESC pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27 : break

        cv2.destroyWindow(self.window_name)



