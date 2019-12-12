import cv2
from glob import glob
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
import sys
from os import listdir

#takes an image and returns array with all chars in it
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
    ret_list =  []

    #for every contour if area large enoug to be digit add the box to list
    li = []
    for cnt in contours:
        if cv2.contourArea(cnt)>20:
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
                cv2.namedWindow('Phuriouz is crazy',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Phuriouz is crazy', 1600,600)
                cv2.imshow('Phuriouz is crazy', im)
                #show for 100 ms and check if exit called (esc key)
                key = cv2.waitKey(0)
                if key == 27:
                    sys.exit()

                #print what the NN would classify the char as
                print(chr(int(classify(samples))), sep=' ', end='', flush=True)
            
            #add char to return list
            ret = ret + chr(int(classify(samples)))
            ret_list.append(ret)

    #return all chars found
    if show_img:
        print('\n')
    return ret_list

#get list of found chars and runs it through NN
def classify(data):
    return int(clf.predict(data))

#check if need to retrain NN, otherwise load old one
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
train = input("Train again? y/n ")
filename = 'finalized_model.sav'

if train == 'y':
    #make NN and train
    clf = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(32), random_state=1, max_iter=5000)
    clf.fit(X_train, Y_train)

    #save the NN
    pickle.dump(clf, open(filename, 'wb'))
else:
    try:
        clf = pickle.load(open(filename, 'rb'))
    except:
        print("No model found")
        sys.exit()

#check if user wants to see image classification animation
show_img = input("Want to show all images being classified? y/n ")
if show_img == 'y':
    show_img = True
else:
    show_img = False

img_mask = 'TestingPictures/*.jpg'
img_names = glob(img_mask)
names = []

#loop over all images
for fn in img_names:
    #read image and zoom in on name
    img = cv2.imread(fn)
    img = img[450:1400, 650:1200]
    img1 = img[30:95, 0:400]
    data = val(img1)
    names.append(data)

    img2 = img[195:260, 0:400]
    data = val(img2)
    names.append(data)

    img3 = img[360:425, 0:400]
    data = val(img3)
    names.append(data)

    img4 = img[515:580, 0:400]
    data = val(img4)
    names.append(data)

    img5 = img[675:740, 0:400]
    data = val(img5)
    names.append(data)

    img6 = img[840:905, 0:400]
    data = val(img6)
    names.append(data)

#format names correctly and print
for i in names:
    names[names.index(i)] = ''.join(str(elem) for elem in i)
print(np.array(names))
