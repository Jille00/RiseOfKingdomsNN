import cv2
from glob import glob
import numpy as np
# import pickle
# from sklearn.preprocessing import normalize
# from sklearn.neural_network import MLPClassifier
import sys
# from os import listdir

import os

# define the name of the directory to be created
a = input("What kingdom? ")
path = "Pictures/" + a

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

img_mask = '*.mp4'
img_names = glob(img_mask)

for i in img_names:
	vidcap = cv2.VideoCapture(i)
	success,image = vidcap.read()
	count = 0
	success = True
	while success:
		# vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*100)) 
		cv2.imwrite(f"{path}/frame%d.jpg" % count, image)     # save frame as JPEG file
		success,image = vidcap.read()
		print ('Read a new frame: ', success)
		count += 1