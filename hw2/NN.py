import math
import numpy as np
import random as rd
import sys
import pickle
def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

## read csv file
trainingData = np.transpose(np.delete(np.transpose(np.genfromtxt(sys.argv[1],delimiter = ",")),0,0))
#valid = trainingData[0:800]
#trainingData = trainingData[800:len(trainingData)]
columnData = np.transpose(trainingData)


y = columnData[57]
#normalize data
'''for i in xrange(0,len(columnData) - 1):
	columnData[i] = (columnData[i] - np.mean(columnData[i])) / np.std(columnData[i]) 
for i in xrange(0,len(columnData[ len(columnData) - 1 ])):
	columnData[len(columnData) - 1][i] = 1
print columnData[len(columnData) - 1]	
trainingData = np.transpose(columnData)'''

## fisrt 58 theta for bias() and 1~57 features and the last for remove the lable
syn0 = 2 * np.random.random((58,64)) - 1
syn1 = 2 * np.random.random((64,1)) - 1
count = 0
for j in xrange(4000):
	l0 = trainingData
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))
	y = y.reshape(y.shape[0],1)
	l2_error = y - l2
	if (j% 100) == 0:
		print "Error:" + str(np.mean(np.abs(l2_error)))
	l2_delta = l2_error*nonlin(l2,deriv=True)
	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error * nonlin(l1,deriv=True)
	syn1 += 0.0001 * l1.T.dot(l2_delta)
	syn0 += 0.0001 * l0.T.dot(l1_delta)



'''
count = 0.0
for i in xrange(0,len(batch[num_of_batch - 1])):
	predict_prob = sig((np.dot(batch[num_of_batch - 1][i],Theta) + bias))
	#print predict_prob
	if predict_prob > 0.5:
		spam = 1
	else :
		spam = 0
	if batch[num_of_batch - 1][i][57] == spam :
		count += 1
print count/500

Theta = np.delete(Theta,len(Theta)-1)

print len(Theta)

testdata =  np.transpose(np.delete(np.transpose(np.genfromtxt(sys.argv[1],delimiter = ",")),0,0))
print len(testdata[0])
test_columnData = np.transpose(testdata)

for i in xrange(0,len(test_columnData)):
	test_columnData[i] = (test_columnData[i] - np.mean(test_columnData[i])) / np.std(test_columnData[i]) 

on = np.ones(len(testdata))
test_columnData = np.vstack((test_columnData,on))
#print test_columnData[len(test_columnData) - 1]	
testdata = np.transpose(test_columnData)
print len(testdata)

predict = []
predict_prob = 0
l0 = testdata
l1 = nonlin(np.dot(l0,syn0))
l2 = nonlin(np.dot(l1,syn1))

for i in xrange(0,len(l2)):
	if(l2[i] > 0.5):
		l2[i] = 1
	else:
		l2[i] = 0
'''
obj = {"syn0": syn0, "syn1": syn1}
pickle.dump(obj,open(sys.argv[2],'w'))


##ok now let's get this fucking logistic reg done



