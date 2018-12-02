# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 23:05:30 2018

@author: paulo
"""

# Standard imports
import cv2
import numpy as np;

# Read image
im = cv2.imread('C:/Users/PauloRenato/Desktop/img3.jpg', cv2.IMREAD_GRAYSCALE)
im = cv2.GaussianBlur(im, (3,3), 1)
im = cv2.Canny(im.copy(),10, 80)
#im = 255-im

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = im.min()
params.maxThreshold = im.max()
params.thresholdStep = 100


# Filter by Area.
#params.filterByArea = True
#params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = True
#params.minCircularity = 0.500
params.minCircularity = 0.7

# Filter by Convexity
#params.filterByConvexity = True
#params.minConvexity = 0.87
    
# Filter by Inertia
#params.filterByInertia = True
#params.minInertiaRatio = 0.01
# Create a detector with the parameters

detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()