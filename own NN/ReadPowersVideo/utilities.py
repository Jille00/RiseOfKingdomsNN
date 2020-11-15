import cv2
import numpy as np

def find_right_sizes(bounded_boxes, thresh, show_image):
    samples =  np.empty((0,100))

    #loop over all digits
    for box in bounded_boxes:
        #unpack data
        x,y,w,h = box[0], box[1], box[2], box[3]

        #check if large enough to be digit but small enough to ignore rest
        if  h>15 and h<30 and w<40 and w>7:

            #draw rectangle with thresh-hold and shape correct form
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            sample = roismall.reshape((1,100))
            samples = np.append(samples,sample,0)

            #if user wants images shown
            if show_image:
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.namedWindow('View Power',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('View Power', 1600,600)
                cv2.imshow('View Power', im)
                #show for 100 ms and check if exit called (esc key)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    sys.exit()
                #print what the NN would classify the digits as
                print(int(classify(samples)))

    return samples

#takes an image and returns array with all digits pixels in it
def extract_digits(image, show_image=False, train=False):
    #copy becuase will be edited
    img = image.copy()

    #make an output, convert to grayscale and apply thresh-hold
    out = np.zeros(image.shape,np.uint8)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    #find conours
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #create empty return list
    
    #for every contour if area large enoug to be digit add the box to list
    li = []
    for cnt in contours:
        if cv2.contourArea(cnt)>20:
            [x,y,w,h] = cv2.boundingRect(cnt)
            li.append([x,y,w,h])

    #sort list so it read from right to left
    li = sorted(li,key=lambda x: x[0], reverse=True)

    #return all digits found
    if not train:
        return find_right_sizes(li, thresh, show_image)
    return find_right_sizes_train(li, thresh, img)

def find_right_sizes_train(bounded_boxes, thresh, im):
    samples =  np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]

    #loop over all digits
    for box in bounded_boxes:
        #unpack data
        x,y,w,h = box[0], box[1], box[2], box[3]

        #check if large enough to be digit but small enough to ignore rest
        if  h>15 and h<30 and w<40 and w>7:

            #draw rectangle with thresh-hold and shape correct form
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),1)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))

            #show digit to user
            cv2.namedWindow('View Power',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('View Power', 600,600)
            cv2.imshow('View Power',im)
            key = cv2.waitKey(0)
            print(key)
            #handle user input
            if key == 27: 
                sys.exit()
            elif key == 108:
                continue
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

    #format and return
    responses = np.array(responses,np.float32)
    return responses, samples


#get list of found digits and runs it through NN
def classify(data, clf):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clas = []

    #run every found digit through NN
    for i in data:
        clas.append(int(clf.predict([i])[0]))

    #reverse list and add all together in 1 integer to find final power
    clas.reverse()
    clas = map(str, clas)
    clas = ''.join(clas)
    return clas