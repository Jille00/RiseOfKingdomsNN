import os
import random
import sys
import warnings
import glob 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from neural_net import CNN_Classifier
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
import pickle

X = np.loadtxt('generalsamples.data',np.float32)
y = np.loadtxt('generalresponses.data',np.float32)
X_norm = normalize(X, axis=1, norm='l2')
tr_ind = int(len(y)*0.8)

#create traiing and testing data
X_train = X_norm[:tr_ind]
X_test = X_norm[tr_ind:]

Y_train = y[:tr_ind]
Y_test = y[tr_ind:]

#make NN and train
clf = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(128, 64), random_state=1, max_iter=5000)
clf.fit(X_train, Y_train)
pickle.dump(clf, open('final_model', 'wb'))


#save the NN

# X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.33, random_state=42)
# lb = LabelEncoder()
# y_train = np_utils.to_categorical(lb.fit_transform(y_train))
# y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# X_train = torch.FloatTensor(X_train)
# X_test = torch.FloatTensor(X_test)
# y_train = torch.LongTensor(y_train)
# y_test = torch.LongTensor(y_test)

# batch_size = 16
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # learning_rate = 0.01
# num_epochs=10

# model = CNN_Classifier().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# train_loss, test_loss = [], []
# train_acc, test_acc = [], []

# for epoch in range(num_epochs):
#     total_loss = 0
#     correct = 0
    
#     #create batches for train data
#     # permutation = torch.randperm(X_train.size()[0])
#     # for i in range(0,X_train.size()[0], batch_size):
#     permutation = torch.randperm(X_train.size()[0])
#     for i in range(0,X_train.size()[0], batch_size):
        
#         #set gradients to zero
#         optimizer.zero_grad()

#         #get correct batches
#         indices = permutation[i:i+batch_size]
#         x, y = X_train[indices], y_train[indices]
#         #cast batches to current device
#         inputs = x.to(device)
#         targets = y.to(device)
#         #forward through network
#         outputs = model(inputs)
        
#         #get loss with ground truth
#         # print(torch.max(outputs, 1)[1], torch.max(targets, 1)[1])
#         loss = criterion(outputs, torch.max(targets, 1)[1])
#         total_loss += loss.item()  
        

#         #calculate number correct
#         # if torch.max(outputs, 1)[1] == torch.max(targets, 1)[1]:
#         #     correct += 1
#         correct += int((torch.max(outputs, 1)[1] == torch.max(targets, 1)[1]).int().sum())
#         # print(torch.max(outputs, 1)[1])
#         # print(torch.max(targets, 1)[1])
        
#         #backprop and optimze
#         loss.backward()
#         optimizer.step()

#     accuracy = 100 * correct / len(X_train)
#     total_loss = total_loss / len(X_train) * batch_size
#     train_acc.append(accuracy)
#     train_loss.append(total_loss)
#     print('Epoch [{}/{}], Train Cross Entropy: {}, Train Accuracy {} '.format(epoch + 1, num_epochs, total_loss,np.round(accuracy,2)))
    
#     with torch.no_grad():
#         total_loss = 0
#         correct = 0
#         permutation = torch.randperm(X_test.size()[0])
#         for i in range(0,X_test.size()[0], batch_size):
            
#             #set gradients to zero
#             optimizer.zero_grad()

#             #get correct batches
#             indices = permutation[i:i+batch_size]
#             x, y = X_test[indices], y_test[indices]
#             inputs = x.to(device)
#             targets = y.to(device)
#             #forward through network
#             outputs = model(inputs)
            
#             #get loss with ground truth
#             loss = criterion(outputs, torch.max(targets, 1)[1])

#             total_loss += loss.item()   
#             correct += int((torch.max(outputs, 1)[1] == torch.max(targets, 1)[1]).int().sum())

#         accuracy = 100 * correct / len(X_test)
#         total_loss = total_loss / len(X_test) * batch_size
#         test_acc.append(accuracy)
#         test_loss.append(total_loss)
#         print('Epoch [{}/{}], Test Cross Entropy: {} , Test Accuracy {}'.format(epoch + 1, num_epochs, total_loss,np.round(accuracy,2)))


# torch.save(model, 'pytorch_model.model')