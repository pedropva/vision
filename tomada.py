# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:32:42 2018

@author: paulo
"""

import cv2, imutils

img = cv2.imread("img4.jpg", 0)

wimg, himg = img.shape[:2]

if(wimg<himg):
    img = imutils.resize(img, width=700)
else:
    img = imutils.resize(img, height=700)

template = cv2.imread("temp.jpg", 0)

w, h = template.shape[::-1]
img = cv2.GaussianBlur(img, (7,7), 1)


res = cv2.matchTemplate(img,template,cv2.TM_SQDIFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img,top_left, bottom_right, 255, 2)
cv2.imshow("res1.jpg", img)
#cv2.imwrite("res1.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()