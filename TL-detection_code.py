# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:04:01 2020

@author: 91742
"""
import cv2
import numpy as np
def adjust_gamma(image, gamma):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / (gamma)
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
    
cap = cv2.VideoCapture(r'C:\Users\91742\Desktop\gamma-correction\v1.mp4')
cv2.namedWindow("gamma value")
def nothing(x):
    pass
cv2.createTrackbar("1-3", "gamma value", 2, 6, nothing)
while True:
    _, frame = cap.read()
    gamma_val = cv2.getTrackbarPos("1-3", "gamma value")
    adjusted = adjust_gamma(frame,gamma_val)
    

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([10,10,200])
    upper_red1 = np.array([25,77,255])
    lower_red2 = np.array([160,10,200])
    upper_red2 = np.array([180,77,255])
    lower_white=np.array([0,10,245])
    upper_white=np.array([180,50,255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3=mask1+mask2
    mask4 = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(mask3,mask4)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(res,kernel,iterations = 3)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    _, contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    """
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if (w/h>1 and w/h<3):
            if(w*h<1500 and w*h>170):
    cv2.imshow('img',np.hstack([frame,adjusted]))
    cv2.imshow('res',res)
    """
    rects = []

    # Bool array indicating which initial bounding rect has
    # already been used
    rectsUsed = []
    
    # Just initialize bounding rects and set all bools to false
    for cnt in contours:
         x, y, w, h = cv2.boundingRect(cnt)
         if( w*h>500):
             rects.append(cv2.boundingRect(cnt))
             rectsUsed.append(False)

    # Sort bounding rects by x coordinate
    def getYFromRect(item):
        return item[1]

    rects.sort(key = getYFromRect)

    # Array of accepted rects
    acceptedRects = []
    
    
    yThr = 5
    

    # Iterate all initial bounding rects
    for supIdx, supVal in enumerate(rects):
        if (rectsUsed[supIdx] == False):

            currxMin = supVal[0]
            currxMax = supVal[0] + supVal[2]
            curryMin = supVal[1]
            curryMax = supVal[1] + supVal[3]

        # This bounding rect is used
            rectsUsed[supIdx] = True

        # Iterate all initial bounding rects
        # starting from the next
            for subIdx, subVal in enumerate(rects[(supIdx+1):], start = (supIdx+1)):

                # Initialize merge candidate
                candxMin = subVal[0]
                candxMax = subVal[0] + subVal[2]
                candyMin = subVal[1]
                candyMax = subVal[1] + subVal[3]
                
            # Check if x distance between current rect
            # and merge candidate is small enough
                if (candyMin-curryMax <= yThr ):
                    if (abs(currxMax-candxMin) or abs(currxMin-candyMax) in [25, 30]):
                        if (subVal[2]*subVal[3]> 3*supVal[2]*supVal[3] or subVal[2]*subVal[3]<6*supVal[2]*supVal[3]):

                # Reset coordinates of current rect
                            currxMax = candxMax
                            curryMin = min(curryMin, candyMin)
                            curryMax = max(curryMax, candyMax)

                # Merge candidate (bounding rect) is used
                            rectsUsed[subIdx] = True
                else:
                    break

        # No more merge candidates possible, accept current rect
            acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])

    for rect in acceptedRects:
     img = cv2.rectangle(adjusted, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)

    cv2.imshow("img", np.hstack([frame,adjusted]))
    cv2.imshow('dil',dilation)
    cv2.imshow('res',res)
    cv2.imshow('morphed',closing)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()