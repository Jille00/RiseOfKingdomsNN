import cv2
from glob import glob
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
import sys
from collections import Counter
import os
import csv
from difflib import SequenceMatcher
import pandas as pd
from time import gmtime, strftime

show_img = False
#takes an image and returns array with all chars in it
def chars_read(im, show=False):
    #copy becuase will be edited
    img = im.copy()

    #make an output, convert to grayscale and apply thresh-hold
    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    #find conours
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    #create empty return list
    ret_list =  []

    #for every contour if area large enoug to be digit add the box to list
    li = []
    for cnt in contours:
        if cv2.contourArea(cnt)>100:
            [x,y,w,h] = cv2.boundingRect(cnt)
            li.append([x,y,w,h])
    #sort list so it read from left to right
    li = sorted(li,key=lambda x: x[0], reverse=False)

    #loop over all chars found
    for i in li:
        #unpack data
        x,y,w,h = i[0], i[1], i[2], i[3]
        #check if large enough to be char but small enough to ignore rest
        if  h>10 and h<40 and w<60:
            ret = ""

            #draw rectangle with thresh-hold and shape to correct form
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            samples =  np.empty((0,100))
            sample = roismall.reshape((1,100))
            samples = np.append(samples,sample,0)

            #if user wants images shown
            if show:
                cv2.namedWindow('Window',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Window', 1600,600)
                cv2.imshow('Window', im)
                #show for 100 ms and check if exit called (esc key)
                key = cv2.waitKey(0)
                if key == 27:
                    sys.exit()

                #print what the NN would classify the char as
                print(chr(int(classify_name(samples))), sep=' ', end='', flush=True)
            
            #add char to return list
            ret = ret + chr(int(classify_name(samples)))
            ret_list.append(ret)

    #return all chars found
    if show_img:
        print('\n')
    return ret_list

def classify_name(data):
    return int(names_model.predict(data))

#list for wrongly classified 
img_list = []

#takes an image and returns array with all digits pixels in it
def digits_read(im, show_img=False, check=False):
    #copy becuase will be edited
    img = im.copy()

    #make an output, convert to grayscale and apply thresh-hold
    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    #find conours
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #create empty return list
    samples =  np.empty((0,100))

    #for every contour if area large enoug to be digit add the box to list
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
        if  h>15 and h<30 and w<40 and w>7:

            #draw rectangle with thresh-hold and shape correct form
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            sample = roismall.reshape((1,100))
            samples = np.append(samples,sample,0)

            #if user wants images shown
            if show_img:
                cv2.namedWindow('View Power',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('View Power', 1600,600)
                cv2.imshow('View Power', im)
                #show for 100 ms and check if exit called (esc key)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    sys.exit()
                #print what the NN would classify the digits as
                print(int(classify(samples)))
    #return all digits found
    #if full number lower than 10m, add to wrongly classified list
    if check:
        try:
            classification = classify(samples)
            if int(classification) < 20000000 and len(classification) > 4 or int(classification) > 400000000:
                #loop over all wrongly classified images and let user enter power manually
                img_list_dict = {48:0, 49:1, 50:2, 51:3, 52:4, 53:5, 54:6, 55:7, 56:8, 57:9}
                add = ""
                while True:
                    cv2.imshow('View Power', img)
                    key = cv2.waitKey(0)
                    back = False
                    if key == 27:
                        sys.exit()
                    elif key == 8:
                        back = True
                    elif key == 13:
                        print('\n')
                        break
                    elif key in img_list_dict.keys():
                        number = img_list_dict[key]

                    if not back:
                        add = add + str(number)
                    else:
                        add = add[:-1]
                    print(add)
                return int(add)
            if len(classification) not in (8,9):
                return False
        except:
            return False
    return samples

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

#get list of found digits and runs it through NN
def classify(data):
    clas = []

    #run every f ound digit through NN
    if isinstance(data, bool):
        return 0
    if isinstance(data, int):
        return data
    for i in data:
        a = int(digits_model.predict([i])[0])
        if a == 11:
            a = 44
        clas.append(a)

    #reverse list and add all together in 1 integer to find final power
    clas.reverse()
    clas = map(str, clas)
    clas = ''.join(clas)
    return clas
#list for wrongly classified 
img_list = []

#load models
try:
	digits_model = pickle.load(open('digits_model.sav', 'rb'))
	names_model = pickle.load(open('names_model.sav', 'rb'))
except:
	print("No models found")
	sys.exit()

# show_img = input("Want to show all images being classified? y/n ")
# if show_img == 'y':
show_img = False
# else:
#     show_img = False

#ask user which kingdom to check
# vs = cv2.VideoCapture('video0.mov')
dirs = os.listdir('TestingPictures/')
for kingdom in dirs:
    # if kingdom == '1359':
    #     show_img=True
    power = []
    path = 'TestingPictures/' + kingdom + '/'
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG") or filename.endswith(".JPG"):

            stats = []

            img = cv2.imread(path + filename)
            # check to see if we have reached the end of the stream
            if img is None:
                break
            img = cv2.resize(img, (1728, 1080))
            y_dim,x_dim,_ = img.shape
            # img = img[int(y/4):y-int(y/12), int(x/1.5):x]
            # cv2.imshow('1', img)
            # cv2.waitKey(0)
            # break
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

            #find conours
            contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            #create empty return list
            samples =  np.empty((0,100))

            #for every contour if area large enoug to be digit add the box to list
            li = []
            for cnt in contours:
                if cv2.contourArea(cnt)>10000 and cv2.contourArea(cnt)<100000:# and cv2.contourArea(cnt)<3000:
                    [x,y,w,h] = cv2.boundingRect(cnt)
                    li.append([x,y,w,h])
            #sort list so it read from right to left
            li = sorted(li,key=lambda x: x[1], reverse=True)
            x,y,w,h = li[-2][0], li[-2][1], li[-2][2], li[-2][3]
            im = img[y:y+h, x+120:x+w-120]
            data = chars_read(im)
            if len(data)>10:
                stats_or_gov = 'gov'
            else:
                stats_or_gov = 'stats'


            if stats_or_gov == 'stats':
                sizes = [int(y_dim/4),y_dim-int(y_dim/12), int(x_dim/1.5),x_dim]
                li = []
                for cnt in contours:
                    if cv2.contourArea(cnt)>1000 and cv2.contourArea(cnt)<5000:
                        [x,y,w,h] = cv2.boundingRect(cnt)
                        li.append([x,y,w,h])
                #sort list so it read from right to left
                li = sorted(li,key=lambda x: x[1], reverse=True)
                for i in li:
                #unpack data
                    x,y,w,h = i[0], i[1], i[2], i[3]
                    if h > 20 and h < 100 and w < 700 and y > sizes[0] and y < sizes[1] and x > sizes[2] and x < sizes[3]:
                        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                        im = img[y-10:y+h+10, x-10:x+w+10]
                        stats.append(int(classify(digits_read(im))))
                # if len(stats) == 6:
                stats = stats[:-1]
                if len(stats) == 7:
                    del stats[-2]

                ###TOP BAR
                try:
                    li = []
                    for cnt in contours:
                        if cv2.contourArea(cnt)>10000 and cv2.contourArea(cnt) < 150000:
                            [x,y,w,h] = cv2.boundingRect(cnt)
                            if h > 100:
                                li.append([x,y,w,h, cv2.contourArea(cnt)])
                    li = sorted(li,key=lambda x: x[4], reverse=True)

                    x,y,w,h,_ = li[-1]
                    sizes = [y-10,y+h+10, x-10,x+w+10]

                    i = []
                    for cnt in contours:
                        if cv2.contourArea(cnt)>1000:
                            [x,y,w,h] = cv2.boundingRect(cnt)
                            li.append([x,y,w,h,cv2.contourArea(cnt)])
                    li = sorted(li,key=lambda x: x[1], reverse=True)
                    li2 = []
                    for i in li:
                        x,y,w,h,c = i

                        if w > 50 and w < 350 and h < 75 and y > sizes[0] and y < sizes[1] and x > sizes[2] and x < sizes[3]:
                            li2.append([x,y,w,h, c])
                            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    li2 = sorted(li2,key=lambda x: x[0], reverse=True)
                    x,y,w,h,_ = li2[2]
                    name = chars_read(img[y-10:y+h+10, x-10:x+w+10])
                    stats.append(''.join(name))

                    x,y,w,h,_ = li2[1]
                    im = img[y-10:y+h+10, x-10:x+w+10]
                    y_dim,x_dim,_ = im.shape
                    im = im[:, x_dim//3:]

                    data = digits_read(im, check=True)
                    data = int(classify(data))
                    stats.append(data)

                    x,y,w,h,_ = li2[0]
                    im = img[y-10:y+h+10, x-10:x+w+10]
                    y_dim,x_dim,_ = im.shape
                    im = im[:, x_dim//3:]
                    data = digits_read(im,check=False)
                    data = int(classify(data))
                    stats.append(data)
                    power.append(stats)

                except:
                    print(filename)




    
    print("Help, Rss ass, Rss gathered, Scout, Dead, Victory, Name, Power, Kills")
    for index, i in enumerate(power):
        if len(i) != 9:
            del power[index] 
    power = np.array(power)
    print(power)


    df = pd.DataFrame(columns=['Name', 'Power', "Kills", 'Dead', 'Rss assistance', 'Help'])

    names = power[:,6]
    powers = power[:,7]
    kills = power[:,8]
    deads = power[:,4]
    rssass = power[:,1]
    helps = power[:,0]

    df['Name'] = names
    df['Power'] = powers
    df['Kills'] = kills
    df['Dead'] = deads
    df['Rss assistance'] = rssass
    df['Help'] = helps

    df.to_excel(path + 'sheet.xlsx')

