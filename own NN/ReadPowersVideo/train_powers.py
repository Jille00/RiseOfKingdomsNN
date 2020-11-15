import cv2
import numpy as np
import pickle
import pandas as pd
import os
import sys
from time import gmtime, strftime
from utilities import *

class extract_data:
	def __init__(self):
		self.dirs = os.listdir('Pictures/')
		self.responses = np.array([])
		self.samples = np.empty((0,100))

	def read_image(self, path):
		img = cv2.imread(path)
		# check to see if we have reached the end of the stream
		if img is None:
			return None

		# img = cv2.resize(img, (1728, 1080))
		y,x,_ = img.shape
		img = img[int(y/4):y-int(y/12), int(x/1.5):x]
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
		kernel = np.ones((2,2), np.uint8)
		dilation = cv2.dilate(thresh, kernel, iterations=1)
		erode = cv2.erode(dilation, kernel, iterations = 2)

		#find conours
		contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
		#create empty return list
		#for every contour if area large enoug to be digit add the box to list
		li = []
		for cnt in contours:
			if cv2.contourArea(cnt)>2000:# and cv2.contourArea(cnt)<3000:
				[x,y,w,h] = cv2.boundingRect(cnt)
				li.append([x,y,w,h])

		#sort list so it read from right to left
		li = sorted(li,key=lambda x: x[0], reverse=True)
		#loop over all digits
		for i in li:
			#unpack data
			x,y,w,h = i[0], i[1], i[2], i[3]

			#check if large enough to be digit but small enough to ignore rest
			if  w<200 and h<50:
				im = img[y-10:y+h+10, x-10:x+w+10]
				responses, samples = extract_digits(im, train=True)
				self.responses = np.append(self.responses, responses, 0)
				self.samples = np.append(self.samples, samples, 0)

	def read_individual_kingdom(self, path, kingdom):
		for filename in os.listdir(path):
			if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG") or filename.endswith(".JPG"):
				self.read_image(path + filename)
				# break
				
	def read_kingdoms(self):
		for kingdom in self.dirs:
			path = 'Pictures/' + kingdom + '/'
			self.read_individual_kingdom(path, kingdom)

		try:
			X = np.loadtxt('generalsamples.data',np.float32)
			Y = np.loadtxt('generalresponses.data',np.float32)
			self.samples = np.concatenate((X, self.amples), axis=0)
			self.responses = np.concatenate((Y, self.responses), axis=0)
		except:
			pass

		np.savetxt('generalsamples.data',self.samples)
		np.savetxt('generalresponses.data',self.responses)

func = extract_data()
func.read_kingdoms()