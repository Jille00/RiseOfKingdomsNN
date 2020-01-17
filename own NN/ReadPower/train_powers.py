import cv2
from glob import glob
import numpy as np
import pickle
import sys

#gather training data for digits NN
def gather_data(im):
	#make copy because will be edited
	im3 = im.copy()

	#convert to grayscale, add blur and apply threshold
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
	#find conours
	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	#create return variables
	samples =  np.empty((0,100))
	responses = []
	keys = [i for i in range(48,58)]

	#find digits
	li = []
	for cnt in contours:
		if cv2.contourArea(cnt)>20:
			[x,y,w,h] = cv2.boundingRect(cnt)
			li.append([x,y,w,h])
			#sort list so it read from right to left
	li = sorted(li,key=lambda x: x[0], reverse=True)

	#loop over all digits
	for i in li:
		#unpack data
		x,y,w,h = i[0], i[1], i[2], i[3]

		#check if large enough to be digit but small enough to ignore rest
		if  h>20 and h<40 and w<40:
		#draw box
			cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
			roi = thresh[y:y+h,x:x+w]
			roismall = cv2.resize(roi,(10,10))

			#show digit to user
			cv2.namedWindow('View Power',cv2.WINDOW_NORMAL)
			cv2.resizeWindow('View Power', 600,600)
			cv2.imshow('View Power',im)
			key = cv2.waitKey(0)

			#handle user input
			if key == 27: 
				sys.exit()
			elif key == 108:
				continue
			if key == 8:
				print("Type in correct number please: ")
				responses.append(11)
				sample = roismall.reshape((1,100))
				samples = np.append(samples, sample, 0)
			elif key in keys:
				responses.append(int(chr(key)))
				sample = roismall.reshape((1,100))
				samples = np.append(samples,sample,0)

	#format and return
	responses = np.array(responses,np.float32)
	return responses, samples

#load pictures
new_text = []
img_mask = 'Pictures/*.jpg'
img_names = glob(img_mask)
text = []
samples =  np.empty((0,100))
responses = []

#loop over all pictures
for fn in img_names:
	img = cv2.imread(fn)
	img = img[450:1400, 1900:2150]
	img1 = img[50:110, 20:220]
	data = gather_data(img1)
	responses = np.append(responses,data[0],0)
	samples = np.append(samples,data[1],0)

	img2 = img[205:265, 20:220]
	data = gather_data(img2)
	responses = np.append(responses,data[0],0)
	samples = np.append(samples,data[1],0)

	img3 = img[370:430, 20:220]
	data = gather_data(img3)
	responses = np.append(responses,data[0],0)
	samples = np.append(samples,data[1],0)

	img4 = img[535:595, 20:220]
	data = gather_data(img4)
	responses = np.append(responses,data[0],0)
	samples = np.append(samples,data[1],0)

	img5 = img[695:755, 20:220]
	data = gather_data(img5)
	responses = np.append(responses,data[0],0)
	samples = np.append(samples,data[1],0)

	img6 = img[860:920, 20:220]
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