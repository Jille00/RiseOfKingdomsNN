import cv2
from glob import glob
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
import sys
from collections import Counter
from os import listdir

show_img = True
def gather_data(im):
	img = im.copy()

	out = np.zeros(im.shape,np.uint8)
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	# thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
	ret,thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
	# cv2.imshow('thresh1', thresh)

	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	samples =  np.empty((0,100))
	responses = []
	keys = [i for i in range(97,123)]
	upper_keys = [i for i in range(65,91)]

	cv2.drawContours(im, contours, -1, (0, 255, 0), 3) 
  
	cv2.imshow('Contours', im) 
	cv2.waitKey(0) 

	li = []
	for cnt in contours:
		if cv2.contourArea(cnt)>175:
			[x,y,w,h] = cv2.boundingRect(cnt)
			li.append([x,y,w,h])
	li = sorted(li,key=lambda x: x[0], reverse=False)

	for i in li:
		x,y,w,h = i[0], i[1], i[2], i[3]
		if  h>20 and h<40 and w<40:
			cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
			roi = thresh[y:y+h,x:x+w]
			roismall = cv2.resize(roi,(10,10))
			cv2.namedWindow('norm',cv2.WINDOW_NORMAL)
			cv2.resizeWindow('norm', 600,600)
			cv2.imshow('norm',im)
			key = cv2.waitKey(0)
			# print(key)

			if key == 27: 
				sys.exit()
			elif key == 92:
				continue
			elif key == 225:
				key = cv2.waitKey(0) - 32
			# elif key == 93:

			if key in keys or key in upper_keys:
				responses.append(int(key))
				sample = roismall.reshape((1,100))
				samples = np.append(samples,sample,0)

	responses = np.array(responses, np.float32)
	return responses, samples

img_mask = 'TestingPictures/*.jpg'
img_names = glob(img_mask)
samples =  np.empty((0,100))
responses = []


#loop over all images
for fn in img_names:
	player = []
	#read image and zoom in on power
	img = cv2.imread(fn)
	img = img[0:1600, 0:2500]

	####CHARS
	name = img[280:380, 620:1000]
	data = gather_data(name)
	responses = np.append(responses,data[0],0)
	samples = np.append(samples,data[1],0)

try:
	X = np.loadtxt('generalsamples.data',np.float32)
	Y = np.loadtxt('generalresponses.data',np.float32)
	samples = np.concatenate((X, samples), axis=0)
	responses = np.concatenate((Y, responses), axis=0)
except:
	pass

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
