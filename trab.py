import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def readBaseMarking():
    marking = []
    pathMarking = os.listdir('./base_tomadas/tipoN')
    for x in range(len(pathMarking)):
        img = cv2.imread("./base_tomadas/tipoN/{arquivo}".format(arquivo = pathMarking[x]),0)
        img = cv2.resize(img,(500,500))
        marking.append(img)
    return marking

def findRectangle(img):
    img2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    nImg = img.copy()
    retFound = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True), True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 2000:
            cv2.drawContours(nImg, [cnt],0, (200, 0, 255), thickness = 5)
            retFound+=1
    
    return nImg,retFound

def rcGama(img,gama):# 0 < gama < 1 ou gama > 1
    rows,cols = img.shape
    img = np.float32(img)
    nvImg = np.zeros((rows,cols),np.uint8).reshape(rows,cols)
    for i in range(rows):
        for j in range(cols):
            nvImg[i,j] = (((img[i,j]*1.0)/255)**gama)*255
    return nvImg

def blobDettection(img):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = img.min()
        params.maxThreshold = img.max()
        params.thresholdStep = 1
        
        
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
        keypoints = detector.detect(img)
        
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob
        
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img_with_keypoints
    
def houghDettection(img):
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,30,
                                param1=50,param2=30,minRadius=0,maxRadius=50)
    
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
    return cimg

def teste():
    ldir = readBaseMarking()
    count = 1
    sucessos = 0
    for img in range(len(ldir)):
        
        img = ldir[img]
        #realce = rcGama(blur,2)
        blur = cv2.GaussianBlur(img, (7,7), 1)
        
        #resultado = cv2.bilateralFilter(blur, 10, 17, 17)
        edges = cv2.Canny(blur, 20, 90)
        
        #img = cv2.pyrMeanShiftFiltering(img, 21, 50)
        #img = cv2.medianBlur(img,5)
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #gray = cv2.bitwise_not(img)
        #edges = cv2.Canny(gray,100,200)
        #th = cv2.threshold(blur, 128, 255, cv2.THRESH_OTSU_INV)[1]
        ret, thresh = cv2.threshold(blur, 128, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        #ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        #TESTAR MORFOLOGIA AMNHA
        #imgResult,rets = findRectangle(edges)
        #imgResult = houghDettection(th)
        #imgResult = blobDettection(gray)
        #if rets > 0: sucessos +=1
        cv2.imshow( "{count}".format(count=count), thresh )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        
        count += 1
        
    #print("Sucessos: {sucessos}".format(sucessos=sucessos))
    return sucessos
if __name__ == '__main__':
    teste()
