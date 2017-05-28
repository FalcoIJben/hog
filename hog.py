# import the necessary packages
from __future__ import print_function
import argparse
import datetime
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--win-stride", type=str, default="(8, 8)",
                help="window stride")
ap.add_argument("-p", "--padding", type=str, default="(16, 16)",
                help="object padding")
ap.add_argument("-s", "--scale", type=float, default=1.05,
                help="image pyramid scale")
ap.add_argument("-m", "--mean-shift", type=int, default=-1,
                help="whether or not mean shift grouping should be used")
args = vars(ap.parse_args())

# evaluate the command line arguments (using the eval function like
# this is not good form, but let's tolerate it for the example)
winStride = eval(args["win_stride"])
padding = eval(args["padding"])
meanShift = True if args["mean_shift"] > 0 else False

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# load the video and make images out of it and resize it
cap = cv2.VideoCapture('/home/falco/Desktop/filmpjes/4.webm')

while (1):
    ret, frame = cap.read()
    if not ret:
        break



    # load the frame and resize it
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    for i in range(0,4):
        crop_frame = frame[0:400, 0+(i*100):200] # Crop from x, y, w, h -> 100, 200, 300, 400

        # detect people in the crop frame
        start = datetime.datetime.now()
        (rects, weights) = hog.detectMultiScale(frame, 0, winStride=winStride,
                                            padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)
        print("[INFO] detection took: {}s".format(
            (datetime.datetime.now() - start).total_seconds()))

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            x = (i * 100) + x
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #cv2.imshow("Detections", crop_frame)

    # show the output image

    cv2.imshow("Crop", frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
