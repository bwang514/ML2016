import math
import numpy as np
import random as rd
import sys
import pickle
def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

a = pickle.load(open(sys.argv[1],'r'))
testdata =  np.transpose(np.delete(np.transpose(np.genfromtxt(sys.argv[2],delimiter = ",")),0,0))
test_columnData = np.transpose(testdata)

for i in xrange(0,len(test_columnData)):
	test_columnData[i] = (test_columnData[i] - np.mean(test_columnData[i])) / np.std(test_columnData[i]) 

on = np.ones(len(testdata))
test_columnData = np.vstack((test_columnData,on))
testdata = np.transpose(test_columnData)

syn0 = a['syn0']
syn1 = a['syn1']

print syn0
l0 = testdata
l1 = nonlin(np.dot(l0,syn0))
l2 = nonlin(np.dot(l1,syn1))

for i in xrange(0,len(l2)):
	if(l2[i] > 0.5):
		l2[i] = 1
	else:
		l2[i] = 0

print(len(l2))
f = open(sys.argv[3], 'w')
f.write("id,label\n")
for i in xrange(0,len(l2)) :
    tmp = str(i+1) + "," + str(int(l2[i])) + "\n"  
    f.write(tmp)
