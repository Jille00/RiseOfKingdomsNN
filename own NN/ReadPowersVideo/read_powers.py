import cv2
from glob import glob
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
import sys
from collections import Counter
from os import listdir

#list for wrongly classified 
img_list = []

#takes an image and returns array with all digits pixels in it
def val(im):
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
        if  h>15 and h<30 and w<40:

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
                key = cv2.waitKey(100)
                if key == 27:
                    sys.exit()
                #print what the NN would classify the digits as
                print(int(classify(samples)))
    #if full number lower than 10m, add to wrongly classified list
    try:
        classification = classify(samples)
        if int(classification) < 10000000 and len(classification) > 4:
            if int(classification) not in img_list:
                img_list.append(img)
            return False
        if len(classification) not in (8,9):
            return False
    except:
        return False

    #return all digits found
    return samples

#get list of found digits and runs it through NN
def classify(data):
    clas = []

    #run every found digit through NN
    for i in data:
        clas.append(int(clf.predict([i])[0]))

    #reverse list and add all together in 1 integer to find final power
    clas.reverse()
    clas = map(str, clas)
    clas = ''.join(clas)
    return clas

#check if need to retrain NN, otherwise load old one
train = input("Train again? y/n ")
filename = 'finalized_model.sav'
if train == 'y':
    #load data and labels
    X = np.loadtxt('generalsamples.data',np.float32)
    Y = np.loadtxt('generalresponses.data',np.float32)

    #normalize and set ratio for training/testing
    X_norm = normalize(X, axis=1, norm='l2')
    tr_ind = int(len(Y)*0.8)

    #create traiing and testing data
    X_train = X_norm[:tr_ind]
    X_test = X_norm[tr_ind:]

    Y_train = Y[:tr_ind]
    Y_test = Y[tr_ind:]

    #make NN and train
    clf = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(32), random_state=1, max_iter=5000)
    clf.fit(X_train, Y_train)

    #save the NN
    pickle.dump(clf, open(filename, 'wb'))
else:
    clf = pickle.load(open(filename, 'rb'))

#check if user wants to see image classification animation
show_img = input("Want to show all images being classified? y/n ")
if show_img == 'y':
    show_img = True
else:
    show_img = False

#ask user which kingdom to check
dirs = listdir('Pictures/')
kingdom = input(f"What kindom would you like to check? {[i for i in dirs]}")
while kingdom not in dirs:
    kingdom = input(f"Please enter correct kingdom? {[i for i in dirs]}")

#get all images in kingdoms subdir
img_mask = f'Pictures/{kingdom}/*.jpg'
img_names = glob(img_mask)
power = []

#loop over all images
for fn in img_names:
    #read image and zoom in on power
    img = cv2.imread(fn)
    img = img[320:950, 1275:1430]
    # cv2.imshow('1', img)
    # cv2.waitKey(0)

    #find all 6 powers in picture and append classified digits to list
    img1 = img[0:70]
    # cv2.imshow('1', img1)
    # cv2.waitKey(0)
    data = val(img1)
    try:
        power.append(int(classify(data)))
    except:
        pass

    img2 = img[110:180]
    # cv2.imshow('1', img2)
    # cv2.waitKey(0)
    data = val(img2)
    try:
        power.append(int(classify(data)))
    except:
        pass

    img3 = img[220:290]
    # cv2.imshow('1', img3)
    # cv2.waitKey(0)
    data = val(img3)
    try:
        power.append(int(classify(data)))
    except:
        pass

    img4 = img[330:400]
    # cv2.imshow('1', img4)
    # cv2.waitKey(0)
    data = val(img4)
    try:
        power.append(int(classify(data)))
    except:
        pass

    img5 = img[440:510]
    # cv2.imshow('1', img5)
    # cv2.waitKey(0)
    data = val(img5)
    try:
        power.append(int(classify(data)))
    except:
        pass    

    img6 = img[550:620]
    # cv2.imshow('1', img6)
    # cv2.waitKey(0)
    data = val(img6)
    try:
        power.append(int(classify(data)))
    except:
        pass

power = list(set(power))
#handle wrongly classified cases 
print(f"Could not read {len(img_list)} numbers. They will be shown to you, type them please!")
print("You can use enter to submit number and backspace to delete and escape to quit")

#loop over all wrongly classified images and let user enter power manually
img_list_dict = {48:0, 49:1, 50:2, 51:3, 52:4, 53:5, 54:6, 55:7, 56:8, 57:9}
for im in img_list: 
    add = ""
    while True:
        cv2.imshow('View Power', im)
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
    power.append(int(add))

#sort whole power list from large to small
sorted_list = sorted(power, reverse=True)
print(f"Length is off by {len(sorted_list)-200} people")
#remove the wrongly classified numbers
# for i in  range(len(img_list)):
#     sorted_list.remove(min(sorted_list))

#print some statistics
total = sum(sorted_list)
top10 = np.sum(sorted_list[0:10])
top25 = np.sum(sorted_list[0:25])
top50 = np.sum(sorted_list[0:50])
top100 = np.sum(sorted_list[0:100])
print(f"Top 10 power is {top10} with average of {top10/10}")
print(f"Top 25 power is {top25} with average of {top25/25}")
print(f"Top 50 power is {top50} with average of {top50/50}")
print(f"Top 100 power is {top100} with average of {top100/100}")
print("Total power", total)
