import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import imutils
import glob

tipo_de_tomada = 'tipoN'

def readBaseMarking():
    marking = []
    pathMarking = os.listdir('./base_tomadas/'+ tipo_de_tomada)
    for x in range(len(pathMarking)):
        img = cv2.imread("./base_tomadas/"+ tipo_de_tomada + "/{arquivo}".format(arquivo = pathMarking[x]),1)
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

def templateMatching(img, angle, templatePath):
    wimg, himg = img.shape[:2]

    if(wimg<himg):
        img = imutils.resize(img, width=700)
    else:
        img = imutils.resize(img, height=700)
    
    template = cv2.imread(templatePath, 0)
    ret, thresh = cv2.threshold(template, 128, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    rotated = imutils.rotate_bound(cv2.bitwise_not(thresh), angle)
    cv2.imshow("Rotated (Correct)", rotated)
    
    
    w, h = template.shape[::-1]
    img = cv2.GaussianBlur(img, (7,7), 1)
    
    res = cv2.matchTemplate(img,rotated,cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img,top_left, bottom_right, 255, 2)
    return img

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


def intrinsicDecomposition(img):
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	#cv2.imshow('original', img)
	#cv2.imshow('hsv', imgHSV)

	V = imgHSV[:,:,2]
	#cv2.imshow('v', V)

	rows, cols = V.shape
	La = np.zeros((rows, cols))
	V = np.float32(V)
	for i in range(rows):
		for j in range(cols):
			La[i,j] = 255*((V[i,j]/255)**(1/2.2))
	V = np.uint8(V)	
	La = np.uint8(La)

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(La)


	#cv2.imshow('Vchannel', V)
	#cv2.imshow('La', La)
	#cv2.imshow('clahe', cl1)
	newHSV = cv2.merge([imgHSV[:,:,0], imgHSV[:,:,1], cl1])
	#cv2.imshow('newhsv', newHSV)
	result = cv2.cvtColor(newHSV, cv2.COLOR_HSV2BGR)

	return result

            
def FODENDOtemplateMatchingAGORAVAITODESESPERADO(img, templatePath,angles,visualize):
    #RETIRADO DE https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
    # load the image image, convert it to grayscale, and detect edges
    template = cv2.imread(templatePath,0)
    #template = cv2.Canny(template, 50, 200)
    bestMatchAngle = 0    
    (tH, tW) = template.shape[:2]
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    found = None
    
    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(img, width = int(img.shape[1] * scale))
        r = img.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or  resized.shape[0] < tW or resized.shape[1] < tW or resized.shape[1] < tH:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        #edged = cv2.Canny(resized, 50, 200)
        for angle in angles: #HERE WE FUCKING TRY TO ROTATE IT TO SEE IF SOMETHING HAPPENS
            rotated = imutils.rotate_bound(template, angle)
            (tH, tW) = rotated.shape[:2]
            result = cv2.matchTemplate(resized, rotated, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    
            # check to see if the iteration should be visualized
            if visualize:
                # draw a bounding box around the detected regionq
                clone = np.dstack([resized, resized, resized])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                    (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv2.imshow("Visualize", clone)
                cv2.imshow("Template", rotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
                bestMatchAngle = angle
                if visualize:
                    print(found,bestMatchAngle)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    print(found,bestMatchAngle)
    rotated = imutils.rotate_bound(template, bestMatchAngle)
    cv2.imshow("Template", rotated)
    (tH, tW) = rotated.shape[:2]
    print(found,bestMatchAngle)
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    # draw a bounding box around the detected result and display the image
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
def teste():
    ldir = readBaseMarking()
    count = 1
    sucessos = 0
    #range(len(ldir))
    for img in range(len(ldir)):
        
        img = ldir[img]
        #realce = rcGama(blur,2)
        blur = cv2.GaussianBlur(img, (7,7), 1)
        
        
        #resultado = cv2.bilateralFilter(blur, 10, 17, 17)
        #edges = cv2.Canny(blur, 20, 90)
        #iluminacaoMelhorada = intrinsicDecomposition(img)
        #img = cv2.pyrMeanShiftFiltering(img, 21, 50)
        #img = cv2.medianBlur(img,5)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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
        
        '''template matching invariante a rotação:
        for angle in [0, 360, 90, 270]:
            imgResult = templateMatching(img,angle,"./base_tomadas/template_"+tipo_de_tomada+".jpg")
            cv2.imshow( "{count}".format(count=count), imgResult)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        '''
        



        #TESTAR O https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html







        
        #template matching invariante a escala:
        imgResult = FODENDOtemplateMatchingAGORAVAITODESESPERADO(gray,"./base_tomadas/template_"+tipo_de_tomada+".jpg",[0,360,270,180],False)
        #cv2.imshow( "{count}".format(count=count), imgResult)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
            
        count += 1
        
    #print("Sucessos: {sucessos}".format(sucessos=sucessos))
    return sucessos
if __name__ == '__main__':
    teste()
