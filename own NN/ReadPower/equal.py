from collections import Counter
import numpy as np

X = np.loadtxt('generalsamples.data',np.float32)
Y = np.loadtxt('generalresponses.data',np.float32)
count = Counter(Y)
li = []
for i in count:
	li.append([i, count[i]])
addition_x = []
addition_y = []
def add_x(y):
	index = 0
	for i in Y:
		if i == y:
			break
		index += 1
	addition_x.append(X[index])

def add_y(y):
	addition_y.append(y)

x_add = []
ma = np.max(li, axis=0)[1]
for i in li:
	while  i[1] < ma:
		add_x(i[0])
		add_y(i[0])
		i[1] += 1

# print(li)
# print(addition_y)
# print(addition_x)

try:
	samples = np.concatenate((X, addition_x), axis=0)
	responses = np.concatenate((Y, addition_y), axis=0)
except:
	pass

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)