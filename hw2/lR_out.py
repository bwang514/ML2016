import numpy as np
import math
import sys
def sig(x):
	if x > 500:
		return 1
	elif  x < -500:
		return 0
	else:	
		return 1 / (1 + math.exp(-x))

model = np.genfromtxt(sys.argv[1],delimiter  = ",")

bias = model[57]
model = np.delete(model,57,0)
testdata =  np.transpose(np.delete(np.transpose(np.genfromtxt(sys.argv[2],delimiter = ",")),0,0))
print len(testdata[0])
test_columnData = np.transpose(testdata)
for i in xrange(0,len(test_columnData)):
	test_columnData[i] = (test_columnData[i] - np.mean(test_columnData[i])) / np.std(test_columnData[i]) 
testdata = np.transpose(test_columnData)
print testdata[56]
predict = []
predict_prob = 0
for i in xrange(0,len(testdata)):
	predict_prob = sig((np.dot(testdata[i],model) + bias))
	print predict_prob
	if predict_prob > 0.5:
		predict.append(1)
	else :
		predict.append(0)

f = open(sys.argv[3], 'w')
f.write("id,label\n")
for i in xrange(0,len(predict)) :
    tmp = str(i+1) + "," + str(predict[i]) + "\n"  
    f.write(tmp)
