import cv2
from glob import glob
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
import sys
from collections import Counter
from os import listdir

#gather character data to train NN on, user inputs label manually
def gather_data(im):
	#make copy because img to be edited
	img = im.copy()

	#make output shape, convert to grayscale and apply thresh-hold
	out = np.zeros(im.shape,np.uint8)
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)

	#find contours
	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	#make outputs
	samples =  np.empty((0,100))
	responses = []
	keys = [i for i in range(97,123)]
	upper_keys = [i for i in range(65,91)]

	#for every contour if area large enoug to be character add the box to list
	li = []
	for cnt in contours:
		if cv2.contourArea(cnt)>175:
			[x,y,w,h] = cv2.boundingRect(cnt)
			li.append([x,y,w,h])
	#sort list so it read from left to right
	li = sorted(li,key=lambda x: x[0], reverse=False)

	#loop over all chars found
	for i in li:
		#unpack data
		x,y,w,h = i[0], i[1], i[2], i[3]

		#check if large enough to be char but small enough to ignore rest
		if  h>20 and h<40 and w<40:

			#draw rectangle with thresh-hold and shape to correct form
			cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
			roi = thresh[y:y+h,x:x+w]
			roismall = cv2.resize(roi,(10,10))

			#show image
			cv2.namedWindow('View Name',cv2.WINDOW_NORMAL)
			cv2.resizeWindow('View Name', 600,600)
			cv2.imshow('View Name',im)
			key = cv2.waitKey(0)

			#check special key cases
			if key == 27: 
				sys.exit()
			elif key == 92:
				continue
			#if char is uppercase
			elif key == 225:
				key = cv2.waitKey(0) - 32

			#add correct label to output
			if key in keys or key in upper_keys:
				responses.append(int(key))
				sample = roismall.reshape((1,100))
				samples = np.append(samples,sample,0)

	#format output and return
	responses = np.array(responses, np.float32)
	return responses, samples

#load images
img_mask = 'Training_Pictures/*.jpg'
img_names = glob(img_mask)
power = []
samples =  np.empty((0,100))
responses = []

#loop over all images
for fn in img_names:
	#read image and zoom in on power
	img = cv2.imread(fn)
	img = img[450:1400, 650:1200]
	img1 = img[20:105, 0:400]
	data = gather_data(img1)
	responses = np.append(responses,data[0],0)
	samples = np.append(samples,data[1],0)

	img2 = img[185:270, 0:400]
	data = gather_data(img2)
	responses = np.append(responses,data[0],0)
	samples = np.append(samples,data[1],0)

	img3 = img[350:435, 0:400]
	data = gather_data(img3)
	responses = np.append(responses,data[0],0)
	samples = np.append(samples,data[1],0)

	img4 = img[505:590, 0:400]
	data = gather_data(img4)
	responses = np.append(responses,data[0],0)
	samples = np.append(samples,data[1],0)

	img5 = img[665:750, 0:400]
	data = gather_data(img5)
	responses = np.append(responses,data[0],0)
	samples = np.append(samples,data[1],0)

	img6 = img[830:915, 0:400]
	data = gather_data(img6)
	responses = np.append(responses,data[0],0)
	samples = np.append(samples,data[1],0)


#save data
try:
	X = np.loadtxt('generalsamples.data',np.float32)
	Y = np.loadtxt('generalresponses.data',np.float32)
	samples = np.concatenate((X, samples), axis=0)
	responses = np.concatenate((Y, responses), axis=0)
except:
	pass

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
