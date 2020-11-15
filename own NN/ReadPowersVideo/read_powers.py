import cv2
import numpy as np
import pickle
import pandas as pd
import os
from glob import glob
from time import gmtime, strftime
from utilities import *
from neural_net import CNN_Classifier
from tkinter import filedialog
from tkinter import *

class extract_data:
    def __init__(self):
        self.filename = 'finalized_model.sav'
        self.clf = clf = pickle.load(open('final_model', 'rb'))
        self.powers = []

    def read_image(self, img):
        # check to see if we have reached the end of the stream
        if img is None:
            return None

        img = cv2.resize(img, (1728, 1080))
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
        samples =  np.empty((0,100))

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
                data = extract_digits(im)
                self.powers.append(int(classify(data, self.clf)))

    def read_individual_kingdom(self, video, path):
        vidcap = cv2.VideoCapture(video)
        success,image = vidcap.read()
        count = 0
        success = True
        while success:
            self.read_image(image)
            success,image = vidcap.read()
            count += 1
                
        #sort whole power list from large to small
        self.powers = list(set(self.powers))
        sorted_list = sorted(self.powers, reverse=True)
        df = pd.DataFrame(columns=['Power', 'Date'])
        sorted_list.append(sum(sorted_list))

        df['Power'] = sorted_list
        df['Index'] = df.index

        df['Index'].iloc[-1] = 'Total'
        # df = df.reset_index(drop=True)
        date = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        df = df[['Index', 'Power', 'Date']]
        df['Date'] = date
        
        # df.index.name = 'Index'
        df.to_excel(path + '_list.xlsx', index=False)


    def main(self):
        ###run gui
        root = Tk()
        root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        path = root.filename.split('/')[:-1]
        path = '/'.join(path) + '/'
        self.read_individual_kingdom(root.filename, path)

func = extract_data()
func.main()
