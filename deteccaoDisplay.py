# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:47:55 2018

@author: Francisco Ylderlan
"""
import numpy as np
import cv2, os

###### Realce do Gama ####################
def rcGama(img,gama):# 0 < gama < 1 ou gama > 1
    rows,cols = img.shape
    img = np.float32(img)
    nvImg = np.zeros((rows,cols),np.uint8).reshape(rows,cols)
    for i in range(rows):
        for j in range(cols):
            nvImg[i,j] = (((img[i,j]*1.0)/255)**gama)*255
    return nvImg

##################### MAIN #################################

fileList = []

fileNames = os.listdir('./Medidores')# lista de nomes dos arquivos de medidores
for x in range(0,len(fileNames)-1,2):
	fileList.append(cv2.imread("./Medidores/{arquivo}".format(arquivo = fileNames[x]),0))#preenchendo lista com os arquivos de imagem


for i in range(len(fileList)):
    img = cv2.resize(fileList[i],(400,600)) # redimensionando
    contraste = rcGama(img,2) # melhorando o brilho da imagem
    resultado = cv2.bilateralFilter(contraste, 10, 17, 17) # aplicando uma filtragem especifica para preservação de borda
    resultado_canny = cv2.Canny(resultado, 20, 235) # algoritmo canny
    #fazendo contorno
    img2,contorno,hierarquia = cv2.findContours(resultado_canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contorno = sorted(contorno, key = cv2.contourArea, reverse = True)[:5]
    for c in contorno:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) == 4 and area > 200:
            janela_contorno = approx
            cv2.drawContours(img,[janela_contorno], 0, (0, 255, 0), 3)
    
    cv2.imshow("Contador "+str(i+1), img)
    cv2.waitKey(0)
    

cv2.destroyAllWindows()






































