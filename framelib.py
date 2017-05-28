import cv2
import settings as s
import copy

def Merge(contours):
    boxxes = []
    rects = []
    for c in contours:
	con = contours[:]
	#TODO
        #if(abs(len(con)-con.index(c)) < 3):
        #    return frame
        #con = con[con.index(c):]

        val = mergeAnyIsAboveRect(c, con)
        if not (val is None):
	    (x, y, w, h) = val
            if ((w * h) < s.minContourSize or h < s.minHeight or w < s.minWidth):
               continue

            rects.insert(len(rects), val)
    
      
    for re in rects:
        r = Box(re)
        found = False
        if(len(boxxes) < 1):
            boxxes.insert(0, r)
            continue
        for b in boxxes:
            if(b.isSameBox(r)):#isAboveRect(b,r)):
                #mergeRect(b, r)
                b.mergeWithBox(r)
                found = True
                continue
        if(not found):
            boxxes.insert(len(boxxes), r)	
                 
    return boxxes



def isAboveBox(box1, box2):
    (x, y, w, h) = cv2.boundingRect(box1)
    (x2, y2, w2, h2) = cv2.boundingRect(box2)

    if((x<=(x2+s.marge) and x+w>=(x2+w2-s.marge)) or (x>=(x2-s.marge) and x+w<=(x2+w2+s.marge))):
	return True
    return False

def mergeAnyIsAboveRect(box, contours):
    for c in contours:
        if(isAboveBox(box, c)):
	    (x, y, w, h) = cv2.boundingRect(box)
            (x2, y2, w2, h2) = cv2.boundingRect(c)
            
            # X as
            if(x > x2):
                if(x+w < x2+w2):
                    w = ((x2+w2)-x)
                x = x2
            else:
                if(x+w < x2+w2):
                    w = ((x2+w2)-x)

            # Y as
            if(y > y2):
                if(y+h < y2+h2):
                    h = ((y2+h2)-y)
                y = y2
            else:
                if(y+h < y2+h2):
                    h = ((y2+h2)-y)

            return (x, y, w, h)
    return





def isAboveRect(box1, box2):
    (x, y, w, h) = box1
    (x2, y2, w2, h2) = box2

    if((x<=(x2+s.marge) and x+w>=(x2+w2-s.marge)) or (x>=(x2-s.marge) and x+w<=(x2+w2+s.marge))):
	return True
    return False

def mergeRect(box, box2):
    (x, y, w, h) = box
    (x2, y2, w2, h2) = box2
    
    # X as
    if(x > x2):
        if(x+w < x2+w2):
            w = ((x2+w2)-x)
        x = x2
    else:
        if(x+w < x2+w2):
            w = ((x2+w2)-x)

    # Y as
    if(y > y2):
        if(y+h < y2+h2):
            h = ((y2+h2)-y)
        y = y2
    else:
        if(y+h < y2+h2):
            h = ((y2+h2)-y)
    return (x, y, w, h)



def mergeLastBox(box, lastBoxxes):
    (x, y, w, h) = box.bounds
    #boxLeft = (x+10, y, w, h)
    #boxRight = (x-10, y, w, h)
    for b in lastBoxxes:
        #box.isSameLastBox(b)
        if(box.isSameLastBox(b) ):#isAboveRect(boxLeft, b) or isAboveRect(boxRight, b) or isAboveRect(box, b)):
            (x2, y2, w2, h2) = b.bounds
            #if(abs((h+y) - (h2+y2)) > 5):
            #    h = h2+y2-y
                
            #if(abs(y - y2) > 5):
            #    h = h + (y - y2)
            #    y = y2
            
            box.bounds = (x, y, w, h)
            box.id = b.id
            return box
    return box



class Box:
    global newID
    global getNewID

    newID = 0
    bounds = (0, 0, 0, 0)
    id = -1

    def __init__(self, bounds):
      global newID
      self.bounds = bounds
      newID += 1
      self.id = newID


    def isSameBox(self, box):
        (x, y, w, h) = self.bounds
        (x2, y2, w2, h2) = box.bounds
        
        
	return ((x<=(x2+s.marge) and x+w>=(x2+w2-s.marge)) or (x>=(x2-s.marge) and x+w<=(x2+w2+s.marge)))
        


    def isSameLastBox(self, lastBox):
        leftLastBox = copy.copy(lastBox)
        rightLastBox = copy.copy(lastBox)

        (x, y, w, h) = leftLastBox.bounds
        x = x - 25
        leftLastBox.bounds = (x, y, w, h)

        (x, y, w, h) = rightLastBox.bounds
        x = x + 25
        rightLastBox.bounds = (x, y, w, h)
        
        return (self.isSameBox(lastBox) or self.isSameBox(leftLastBox) or self.isSameBox(rightLastBox))
    
    def mergeWithBox(self, box):
        (x, y, w, h) = self.bounds
        (x2, y2, w2, h2) = box.bounds
    
        # X as
        if(x > x2):
            if(x+w < x2+w2):
                w = ((x2+w2)-x)
            x = x2
        else:
            if(x+w < x2+w2):
                w = ((x2+w2)-x)

        # Y as
        if(y > y2):
            if(y+h < y2+h2):
                h = ((y2+h2)-y)
            y = y2
        else:
            if(y+h < y2+h2):
                h = ((y2+h2)-y)
        self.bounds =  (x, y, w, h)
        
        #TODO verzin hier iets slims voor, en in de init
        if(self.id > box.id):
            self.id = box.id
        
  


        












