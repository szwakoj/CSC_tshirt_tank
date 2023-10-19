import sys
import numpy as np
import tensorflow as tf
import cv2
import dlib

CROSSHAIR_LEN = 4

#Draws crosshairs centered on (x,y)
def draw_crosshair(x, y, img):
    cv2.line(img, (x+CROSSHAIR_LEN, y), (x-CROSSHAIR_LEN, y), (0,255,0), 1)
    cv2.line(img, (x, y+CROSSHAIR_LEN), (x, y-CROSSHAIR_LEN), (0,255,0), 1)

def main():
    #Check versions and make sure they are installed properly
    print("Tensorflow:", tf.__version__)
    print("Numpy:", np.__version__)
    print("OpenCV:", cv2.__version__)

    #grabbing default webcam (second one is a droidCam IP)
    print("Opening webcam...")
    #camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera = cv2.VideoCapture('http://10.0.97.247:4747/mjpegfeed?640x480')

    #checks to make sure camera was found
    if not camera.isOpened():
        print("ERROR: Cannot open camera!")
        exit(1)

    #open our cv window
    cv2.namedWindow("tank_view", cv2.WINDOW_AUTOSIZE)

    #load a default open cv face detection algorithm
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #initialize out lists and some counters
    frame_num = 0
    current_targetID = 0
    bounding_boxes = []
    trackers = {}

    #while the key hasnt been pressed
    key = -1
    while key == -1:
        _, frame = camera.read()

        #run through the current trackers on hand to check their quality
        #remove those deemed poopy
        bad_targets = []
        for id in trackers.keys():
            quality = trackers[id].update(frame)
            if quality < 7:
                bad_targets.append(id)
                print("Target " + str(id) + " out of view.")

        #reset target numbers
        for id in bad_targets:
            trackers.pop(id, None)
            current_targetID = len(trackers)

        #every tenth frame
        if frame_num % 10 == 0:
            #detect faces and extract bounding boxes
            bounding_boxes = haar_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=6)
            for (_x, _y, _w, _h) in bounding_boxes:
                #convert to int for dlib
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                mid_x = int(x + (w * 0.5))
                mid_y = int(y + (h * 0.5))

                matched_id = None
                #check to see which tracker the current face is in, aswell as which tracker is within the current face
                for id in trackers.keys():
                    position = trackers[id].get_position()

                    t_x = int(position.left())
                    t_y = int(position.top())
                    t_w = int(position.width())
                    t_h = int(position.height())

                    t_mid_x = t_x + (t_w // 2)
                    t_mid_y = t_y + (t_h // 2)

                    if ( ( t_x <= mid_x <= (t_x + t_w)) and
                         ( t_y <= mid_y <= (t_y + t_h)) and
                         ( x <= t_mid_x <= (x + h)) and
                         ( y <= t_mid_y <= (y + h))):

                        matched_id = id
                # if face has no match to current trackers create new tracker
                if matched_id is None:
                    print("New target found, id " + str(current_targetID))
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(frame, dlib.rectangle(x, y, x+w, y+h))
                    trackers[current_targetID] = tracker

                    current_targetID += 1

        #draw bounding boxes around the trackers and crosshairs on center and chest
        for id in trackers.keys():
            position = trackers[id].get_position()
            x = int(position.left())
            y = int(position.top())
            w = int(position.width())
            h = int(position.height())
            mid_x = x + (w // 2)
            mid_y = y + (h // 2)

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)
            draw_crosshair(mid_x, mid_y, frame)
            if y + h + CROSSHAIR_LEN < frame.shape[0]:
                draw_crosshair(mid_x, y + (h * 2), frame)

            cv2.putText(frame, "target_"+str(id), ((x + w), y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)

        """for (x, y, w, h) in bounding_boxes:
            mid_x = int(x + (w // 2))
            mid_y = int(y + (h // 2))
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)
            draw_crosshair(mid_x, mid_y, frame)
            #draw cross hair for chest aiming
            if y + h + CROSSHAIR_LEN < frame.shape[0]:
                draw_crosshair(mid_x, y + (h * 2), frame)"""

        cv2.imshow("tank_view", frame)

        frame_num += 1
        key = cv2.waitKey(10)

    camera.release()
    cv2.destroyAllWindows()

    print("Closing application...")

if __name__ == "__main__":
    main()
