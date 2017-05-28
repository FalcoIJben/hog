import numpy as np
import cv2
import matplotlib
import settings as s
import framelib as f

body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

cap = cv2.VideoCapture('/home/falco/Desktop/filmpjes/4.webm')
fgbg = cv2.createBackgroundSubtractorMOG2()

font = cv2.FONT_HERSHEY_SIMPLEX

lastBoxxes = []

while(1):
    copyLastBoxxes = True
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)


    kernel = np.ones((s.kernel1X, s.kernel1Y), np.uint8)
    kernel2 = np.ones((s.kernel2X, s.kernel2Y), np.uint8)

    
    erosion = cv2.erode(fgmask, kernel, s.kernel1_iteration)

    #kernel2 = np.ones((s.kernel2X, s.kernel2Y), np.uint8)
    dilation = cv2.dilate(erosion, kernel2, s.kernel2_iteration)

    normalized = cv2.normalize(dilation, dilation, 0, 255, cv2.NORM_MINMAX)

    ret,thresh = cv2.threshold(dilation,127,255,cv2.THRESH_TOZERO)
    
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, s.contour_index, (s.countour_R, s.countour_G, s.countour_B), 5)

    #for c in contours:
    #	if cv2.contourArea(c) < s.minContourSize:
    #      continue
    
    #	(x, y, w, h) = cv2.boundingRect(c)
    #   cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    boxxes = f.Merge(contours)

    if(not ((len(lastBoxxes) - len(boxxes)) < 1)):
        print(len(boxxes))

    if((abs(len(lastBoxxes) - len(boxxes)) > len(boxxes)) and len(lastBoxxes) > len(boxxes)):
        boxxes = lastBoxxes 
        lastBoxxes = [] 
        copyLastBoxxes = False 
             

    for b in boxxes:
        box = f.mergeLastBox(b, lastBoxxes)
        (x, y, w, h) = b.bounds
        #move this logic
        if(w-s.maxWidth > 0):
            x = x + int((w-s.maxWidth)/2) 
            w = s.maxWidth
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #TODO
        cv2.putText(frame,"{}".format(b.id), (x, y + 30), font, 1,(255, 0, 0),2)
        
        
        #lastBoxxes.insert(len(lastBoxxes), (x, y, w, h))
    
    

    if(copyLastBoxxes):
        lastBoxxes = []
        for b in boxxes:
            lastBoxxes.insert(len(lastBoxxes), b)
    
    cv2.imshow('frame', dilation)
    cv2.imshow('frame1', frame)

    

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()


