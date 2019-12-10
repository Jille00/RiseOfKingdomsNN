import cv2
from glob import glob
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
import sys
from collections import Counter

img_list = []

def val(im):
    img = im.copy()
    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    samples =  np.empty((0,100))
    li = []
    for cnt in contours:
        if cv2.contourArea(cnt)>20:
            [x,y,w,h] = cv2.boundingRect(cnt)
            li.append([x,y,w,h])
    li = sorted(li,key=lambda x: x[0], reverse=True)
    for i in li:
        x,y,w,h = i[0], i[1], i[2], i[3]
        if  h>20 and h<40 and w<40:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            sample = roismall.reshape((1,100))
            samples = np.append(samples,sample,0)
    if int(classify(samples)) < 10000000:
        img_list.append(img)
    return samples

X = np.loadtxt('generalsamples.data',np.float32)
Y = np.loadtxt('generalresponses.data',np.float32)
X_norm = normalize(X, axis=1, norm='l2')
tr_ind = int(len(Y)*0.8)

X_train = X_norm[:tr_ind]
X_test = X_norm[tr_ind:]

Y_train = Y[:tr_ind]
Y_test = Y[tr_ind:]

train = input("Train again? ")
filename = 'finalized_model.sav'
if train == 'y':
    clf = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(32), random_state=1, max_iter=5000)
    clf.fit(X_train, Y_train)
    pickle.dump(clf, open(filename, 'wb'))
else:
    clf = pickle.load(open(filename, 'rb'))

def classify(data):
    clas = []
    for i in data:
        clas.append(int(clf.predict([i])[0]))
    clas.reverse()
    clas = map(str, clas)
    clas = ''.join(clas)
    return clas

img_mask = 'TestPictures/*.jpg'
img_names = glob(img_mask)
power = []
for fn in img_names:
    img = cv2.imread(fn)
    img = img[450:1400, 1900:2150]
    img1 = img[40:110, 20:220]
    data = val(img1)
    power.append(int(classify(data)))

    img2 = img[195:265, 20:220]
    data = val(img2)
    power.append(int(classify(data)))

    img3 = img[360:430, 20:220]
    data = val(img3)
    power.append(int(classify(data)))

    img4 = img[525:595, 20:220]
    data = val(img4)
    power.append(int(classify(data)))

    img5 = img[685:755, 20:220]
    data = val(img5)
    power.append(int(classify(data)))

    img6 = img[850:920, 20:220]
    data = val(img6)
    power.append(int(classify(data)))

print(f"Could not read {len(img_list)} numbers. They will be shown to you, type them please!")
print("You can use enter to submit number and backspace to delete and escape to quit")
for im in img_list: 
    add = ""
    while True:
        cv2.imshow('1', im)
        key = cv2.waitKey(0)
        back = False
        if key == 27:
            sys.exit()
        elif key == 8:
            back = True
        elif key == 13:
            break
        elif key == 48:
            number = 0
        elif key == 49:
            number = 1
        elif key == 50:
            number = 2
        elif key == 51:
            number = 3
        elif key == 52:
            number = 4
        elif key == 53:
            number = 5
        elif key == 54:
            number = 6
        elif key == 55:
            number = 7
        elif key == 56:
            number = 8
        elif key == 57:
            number = 9

        if not back:
            add = add + str(number)
        else:
            add = add[:-1]
        print(add)
    power.append(int(add))

sorted_list = sorted(power, reverse=True)
for i in  range(len(img_list)):
    sorted_list.remove(min(sorted_list))

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