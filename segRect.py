# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:24:42 2018

@author: paulo
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2, os

def findMedidor(img):
    blur = cv2.GaussianBlur(img, (7,7), 1)
    edges = cv2.Canny(blur, 70, 90)
    img2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    nImg = img.copy()
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True), True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 2000:
            cv2.drawContours(nImg, [cnt],0, (200, 0, 255), thickness = 5)
    
    return nImg

########################MAIN######################################
    
medidoresPath = 'Medidores'
markedPath = 'Marked'

if not os.path.exists(markedPath):
    os.makedirs(markedPath)

files = os.listdir(medidoresPath)

for file in files:
    extension = (os.path.splitext(file))[1]
    if extension == '.jpg' or extension == '.png':
        img = cv2.imread(medidoresPath+'/'+file)
        markedImage = findMedidor(img)
        cv2.imwrite(markedPath+'/'+file, markedImage)