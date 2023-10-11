import sys
import numpy as np
import tensorflow as tf
import cv2

CROSSHAIR_LEN = 4

def draw_crosshair(x, y, img):
    cv2.line(img, (x+CROSSHAIR_LEN, y), (x-CROSSHAIR_LEN, y), (0,255,0), 1)
    cv2.line(img, (x, y+CROSSHAIR_LEN), (x, y-CROSSHAIR_LEN), (0,255,0), 1)

def main():

    print("Tensorflow:", tf.__version__)
    print("Numpy:", np.__version__)
    print("OpenCV:", cv2.__version__)

    print("Opening webcam...")

    camera = cv2.VideoCapture('http://10.0.97.247:4747/mjpegfeed?640x480')

    if not camera.isOpened():
        print("ERROR: Cannot open camera!")
        exit(1)

    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)

    frame_num = 0
    key = -1
    while key == -1:
        _, frame = camera.read()

        if frame_num % 5 == 0:
            grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            bounding_boxes = haar_cascade.detectMultiScale(grey_frame, scaleFactor=1.1, minNeighbors=9)

        for (x, y, w, h) in bounding_boxes:
            mid_x = x + (w // 2)
            mid_y = y + (h // 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)
            draw_crosshair(mid_x, mid_y, frame)
            #draw cross hair for chest aiming
            if y + h + CROSSHAIR_LEN < frame.shape[0]:
                draw_crosshair(mid_x, y + (h * 2), frame)

        cv2.imshow("base-image", frame)



        frame_num += 1
        key = cv2.waitKey(10)

    camera.release()
    cv2.destroyAllWindows()

    print("Closing application...")




if __name__ == "__main__":
    main()
