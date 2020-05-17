import cv2
from glob import glob
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
import sys
from collections import Counter
from os import listdir
import csv
from difflib import SequenceMatcher
show_img = False
#takes an image and returns array with all chars in it
def chars_read(im):
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
        if  h>20 and h<40 and w<60:
            ret = ""

            #draw rectangle with thresh-hold and shape to correct form
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            samples =  np.empty((0,100))
            sample = roismall.reshape((1,100))
            samples = np.append(samples,sample,0)

            #if user wants images shown
            if show_img:
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
def digits_read(im, check=False):
    #copy becuase will be edited
    img = im.copy()

    #make an output, convert to grayscale and apply thresh-hold
    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)

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

        #cehck if large enough to be digit but small enough to ignore rest
        if  h>20 and h<40 and w<40:

            #draw rectangle with thresh-hold and shape correct form
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            sample = roismall.reshape((1,100))
            samples = np.append(samples,sample,0)

            #if user wants images shown
            if show_img:
                cv2.namedWindow('1',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('1', 1600,600)
                cv2.imshow('1', im)
                #show for 100 ms and check if exit called (esc key)
                key = cv2.waitKey(0)
                if key == 27:
                    sys.exit()
                #print what the NN would classify the digits as
                print(int(classify(samples)))
    
    #if full number lower than 10m, add to wrongly classified list
    if check == True and int(classify(samples)) < 10000000:
        img_list.append([im, int(classify(samples))])
    #return all digits found
    return samples

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

#get list of found digits and runs it through NN
def classify(data):
    clas = []

    #run every found digit through NN
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

show_img = input("Want to show all images being classified? y/n ")
if show_img == 'y':
    show_img = True
else:
    show_img = False

dirs = listdir('TestingPictures/')

#get all images in kingdoms subdir
players = []
list_active = False
old_list_names = []
#loop over all images
for j in dirs:
    img_mask = f'TestingPictures/{j}/*.jpg'
    img_names = glob(img_mask)
    for fn in img_names:
        player = []

        #read image and zoom in on power
        img = cv2.imread(fn)
        img = img[0:1600, 0:2500]

        ####CHARS
        name = img[260:400, 600:1100]
        data = chars_read(name)
        data = ''.join(str(elem) for elem in data)
        player.append(data)

        ####DIGITS	
        power = img[270:370, 1300:1700]
        data = digits_read(power, True)
        data = int(classify(data))
        player.append(data)

        kills = img[270:390, 1835:2250]
        data = digits_read(kills)
        data = int(classify(data))
        player.append(data)

        victories = img[570:670, 1930:2110]
        data = digits_read(victories)
        data = int(classify(data))
        player.append(data)

        dead = img[770:860, 1900:2150]
        data = digits_read(dead)
        data = int(classify(data))
        player.append(data)

        rss_ass = img[1130:1240, 1800:2100]
        data = digits_read(rss_ass)
        data = int(classify(data))
        player.append(data)

        alliance_help = img[1250:1330, 1850:2100]
        data = digits_read(alliance_help)
        data = int(classify(data))
        player.append(data)

        if j != 'new':
            players.append(player)
        else:
            no = True
            if list_active == False:
                old_list_names = [i[0] for i in players]
                list_active = True
            for i in players:
                sim = similar(i[0].lower(),player[0].lower())
                # if sim > 0.5:
                    # print(i[0], player[0], sim)
                if sim > 0.9:
                    a = player[1:]
                    for b in a:
                        i.append(b)
                    no = False
            if no: 
                if player[0] not in old_list_names:
                    for _ in range(0,6):
                        player.insert(1,0)
                    players.append(player)
                


players.insert(0,['Player name', 'Power (old)', 'Kills (old)', 'Victories (old)', 'Dead (old)', 'Rss-assistance (old)', 'Alliance help (old)', 'Power (new)', 'Kills (new)', 'Victories (new)', 'Dead (new)', 'Rss-assistance (new)', 'Alliance help (new)'])
#handle wrongly classified cases 
print(f"Could not read {len(img_list)} numbers. They will be shown to you, type them please!")
print("You can use enter to submit number and backspace to delete and escape to quit")

#loop over all wrongly classified images and let user enters manually
img_list_dict = {48:0, 49:1, 50:2, 51:3, 52:4, 53:5, 54:6, 55:7, 56:8, 57:9}
for it in img_list: 
    im = it[0]
    add = ""
    while True:
        cv2.imshow('View Digits', im)
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
    for i in players:
    	try:
    		a = i.index(it[1])
    		i[a] = int(add)
    	except:
    		pass

#save data as csv file
with open('total_stats.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in players:
    	wr.writerow(i)
