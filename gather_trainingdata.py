import cv2
from glob import glob
import numpy as np
import pickle
import sys

def gather_data(im):
	im3 = im.copy()

	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	img2 = cv2.drawContours(im3, contours, -1, (0,255,0), 3)

	samples =  np.empty((0,100))
	responses = []
	keys = [i for i in range(48,58)]

	for cnt in contours:
		if cv2.contourArea(cnt)>20:
			[x,y,w,h] = cv2.boundingRect(cnt)

			if  h>20:
				cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
				roi = thresh[y:y+h,x:x+w]
				roismall = cv2.resize(roi,(10,10))
				cv2.namedWindow('norm',cv2.WINDOW_NORMAL)
				cv2.resizeWindow('norm', 600,600)
				cv2.imshow('norm',im)
				key = cv2.waitKey(0)

				if key == 27: 
					sys.exit()
				elif key == 108:
					continue
				elif key in keys:
					responses.append(int(chr(key)))
					sample = roismall.reshape((1,100))
					samples = np.append(samples,sample,0)

	responses = np.array(responses,np.float32)
	return responses, samples

new_text = []
img_mask = 'TestPictures/*.jpg'
img_names = glob(img_mask)
text = []
samples =  np.empty((0,100))
responses = []
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

#UNCOMMENT TO SAVE DATA
# responses = responses.reshape((responses.size,1))
# np.savetxt('generalsamples.data',samples)
# np.savetxt('generalresponses.data',responses)