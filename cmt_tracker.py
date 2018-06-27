import time

import cv2
import numpy as np
from CMT import CMT, util


class CMT_Tracker:

    def __init__(self, window_name, net, frames, kalman, controller, tol):
        
        self.window_name = window_name
        self.tol = tol
        self.frames = frames
        self.kalman = kalman
        self.controller = controller

    def locate(self, frame):


        #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame , 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        found_persons = []

        # loop over the detections
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

    def track_object(self, frame, out, rect):

        tracking_height = rect[3]

        # Convert to point representation, adding singleton dimension
        bbox = util.bb2pts(rect[None, :])

        # Squeeze
        bbox = bbox[0, :]

        tl = bbox[:2]
        br = bbox[2:4]

        CMT.initialise(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), tl, br)

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

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                CMT.process_frame(frame_gray)

                elapsed_bbox = time.time() - t0

                elapsed_frame = time.time() - t1

                if CMT.has_result:

                    # Start timer
                    timer = cv2.getTickCount()

                    # Calculate Frames per second (FPS)
                    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

                    if elapsed_frame > 0.05:

                        px = CMT.tl[0] + int(CMT.br[0] - CMT.tl[0])/2
                        py = CMT.tl[1] + int(CMT.br[1] - CMT.tl[1])/2

                        px_filtered, py_filtered = np.round(self.kalman.filter_pixel((px, py), t)[:2]).astype(np.int)
                        
                        ex = px_filtered - frame.shape[1]/2
                        ey = py_filtered - frame.shape[0]/2

                        self.controller.set_velocities(ex, prev_ex, ey, prev_ey, tracking_height - height, prev_ez, 0, 0)

                        self.controller.move_drone()

                        prev_ez = tracking_height - height
                        prev_ex = ex
                        prev_ey = ey
                        t1 = time.time()

                    util.draw_keypoints(CMT.tracked_keypoints, frame, (255, 255, 255))
                    # this is from simplescale
                    util.draw_keypoints(CMT.votes[:, :2], frame)  # blue
                    util.draw_keypoints(CMT.outliers[:, :2], frame, (0, 0, 255))

                    # Tracking bounding box
                    cv2.rectangle(frame, CMT.tl, CMT.br, (255,0,0), 2, 1)

                else :      
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

                    self.controller.stop_drone()
                    failure += 1
                    if failure > 50:
                        return False

                # Display tracker type on frame
                cv2.putText(frame, "CMT Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

                # Display FPS on frame
                cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

                # Display result
                frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=3, fy=3), cv2.COLOR_RGB2BGR)
                cv2.imshow(self.window_name, frame) 
                #out.write(frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break

        return True

    def track(self, out):

        cv2.namedWindow(self.window_name)

        stop = False

        while not stop:

            # Obtain new frame
            frame, t = self.frames.next()

            if frame is not None:

                found_persons = self.locate(frame)

                if found_persons:
                    # TODO: IF MORE THAN ONE PERSON IS FOUND, CHECK FOR SIMILARITIES BETWEEN THE TRACKING
                    (startX, startY, endX, endY) = found_persons[0].astype(int)

                    # Detection bounding box
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2, 1)

                    rect = np.array([startX, startY, endX - startX, endY - startY])

                    stop = self.track_object(frame, out, rect)

                frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=3, fy=3), cv2.COLOR_RGB2BGR)
                cv2.imshow(self.window_name, frame)

                # Exit if ESC pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27 : break

        cv2.destroyWindow(self.window_name)

