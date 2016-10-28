import math
import numpy as np
import random as rd
import sys

def sigmoid(x):
	siglist = []
	for i in xrange(0,len(x)):
	#	print x[i]
		if x[i] > 500:
			siglist.append(1)
		elif  x[i] < -500:
			siglist.append(0)
		else:	
			siglist.append(1 / (1 + math.exp(-x[i])))
	return np.array(siglist)
def sig(x):
	if x > 500:
		return 1
	elif  x < -500:
		return 0
	else:	
		return 1 / (1 + math.exp(-x))
## read csv file
trainingData = np.transpose(np.delete(np.transpose(np.genfromtxt(sys.argv[1],delimiter = ",")),0,0))
#valid = trainingData[0:800]
#trainingData = trainingData[800:len(trainingData)]
columnData = np.transpose(trainingData)

#normalize data
for i in xrange(0,len(columnData) - 1):
	columnData[i] = (columnData[i] - np.mean(columnData[i])) / np.std(columnData[i]) 

trainingData = np.transpose(columnData)
#spliting batch data
batch = []
col_batch = []
for i in xrange(0, (len(trainingData) / 500)):
	tmp = trainingData[i * 500 : (i+1) * 500] 
	batch.append(tmp)
	col_batch.append(np.transpose(tmp))

num_of_batch = len(batch)
featuresNum = 57
Theta = []
for i in xrange(0,featuresNum):
    Theta.append(rd.uniform(-0.1,0.1))

Theta.append(0)
Theta = np.array(Theta)
bias = rd.uniform(-0.1,0.1)
Learning_rate = 0.01
iterations = 50000
AdaGrad = np.zeros(len(Theta) + 1)
## fisrt 58 theta for bias() and 1~57 features and the last for remove the lable

#start log_reg
for i in xrange(0,iterations):
	batch_it = i % (num_of_batch)
	print i 
	probability = sigmoid(np.dot(batch[batch_it],Theta) + bias)
	difference = -1 * (col_batch[batch_it][57] - probability)
	grad_map = col_batch[batch_it] * difference
	## first deal with bias
	b_grad = np.sum(difference)
	AdaGrad[0] += b_grad**2
	
	bias = bias - (Learning_rate/(AdaGrad[0]**(1/2))) * (b_grad)
#	bias = bias - (Learning_rate) * (b_grad)
	for j in xrange(0,len(grad_map) - 1):
		#print j 
		w_grad = np.sum(grad_map[j])
		AdaGrad[j+1] += w_grad**2
		Theta[j] = Theta[j] - (Learning_rate / (AdaGrad[j+1] ** (1/2))) * (w_grad)
		#Theta[j] = Theta[j] - (Learning_rate) * (w_grad)
	

#testing example

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

f = open(sys.argv[2], 'w')
for i in xrange(0,len(Theta)) :
	tmp = str(Theta[i]) + ',' 
	f.write(tmp)   
f.write(str(bias))


##ok now let's get this fucking logistic reg done



