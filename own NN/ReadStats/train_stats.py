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
	digits = [i for i in range(48, 58)]

	#for every contour if area large enoug to be character add the box to list
	li = []
	for cnt in contours:
		if cv2.contourArea(cnt)>125:
			[x,y,w,h] = cv2.boundingRect(cnt)
			li.append([x,y,w,h])
	#sort list so it read from left to right
	li = sorted(li,key=lambda x: x[0], reverse=False)

	#loop over all chars found
	for i in li:
		#unpack data
		x,y,w,h = i[0], i[1], i[2], i[3]

		#check if large enough to be char but small enough to ignore rest
		if  h>20 and h<40 and w<60 and w>8:

			#draw rectangle with thresh-hold and shape to correct form
			cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
			roi = thresh[y:y+h,x:x+w]
			roismall = cv2.resize(roi,(10,10))

			#show image
			cv2.namedWindow('View Stats',cv2.WINDOW_NORMAL)
			cv2.resizeWindow('View Stats', 600,600)
			cv2.imshow('View Stats',im)
			key = cv2.waitKey(0)
			# print(key)
			#check special key cases
			if key == 8:
				samples = samples[:-1]
				responses = responses[:-1]
				key = cv2.waitKey(0)
			if key == 27: 
				sys.exit()
			elif key == 92:
				continue
			#if char is uppercase
			if key == 225:
				key = cv2.waitKey(0) - 32

			#add correct label to output
			if key in keys or key in upper_keys or key in digits:
				responses.append(int(key))
				sample = roismall.reshape((1,100))
				samples = np.append(samples,sample,0)

	#format output and return
	responses = np.array(responses, np.float32)
	return responses, samples

#get images
img_mask = 'TestingPictures/Train/*.jpg'
img_names = glob(img_mask)
samples =  np.empty((0,100))
responses = []


#loop over all images
for fn in img_names:
	print(len(img_names) - img_names.index(fn))
	player = []
	#read image and zoom in on power
	img = cv2.imread(fn)
	img = img[0:1600, 0:2500]

	####CHARS
	name = img[280:380, 620:1000]
	data = gather_data(name)
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