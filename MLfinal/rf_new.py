import sys
import random
import copy
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score




inputdata = []
traindata = []
attacktypes = ['dos', 'u2r', 'r2l', 'probe']
labeldict = dict()
labeldict['normal'] = 0

with open('train', 'r') as trainfile:
	tmp = trainfile.readlines()
	for i in range(len(tmp)):
		inputdata.append(tmp[i])

featuresize = 41
datasize = len(inputdata)
print datasize

for i in range(datasize):
	traindata.append(inputdata[i].split('.\n')[0].split(','))

with open('training_attack_types.txt', 'r') as typesfile:
	tmp = typesfile.readlines()
	for i in range(len(tmp)):
		tmp_type = tmp[i].split('\n')[0].split(' ')
		for j in range(len(attacktypes)):
			if (tmp_type[1] == attacktypes[j]):
				labeldict[tmp_type[0]] = j+1

print labeldict

type1, count1 = [], 0
type2, count2 = [], 0
type3, count3 = [], 0

onehot_1 = []
onehot_2 = []
onehot_3 = []



for i in range(datasize):
	if (traindata[i][1] not in type1):
		type1.append(traindata[i][1])
		traindata[i][1] = count1
		count1 += 1
	else:
		traindata[i][1] = type1.index(traindata[i][1])
	if (traindata[i][2] not in type2):
		type2.append(traindata[i][2])
		traindata[i][2] = count2
		count2 += 1
	else:
		traindata[i][2] = type2.index(traindata[i][2])
	if (traindata[i][3] not in type3):
		type3.append(traindata[i][3])
		traindata[i][3] = count3
		count3 += 1
	else:
		traindata[i][3] = type3.index(traindata[i][3])


for i in range(count1):
	a = [0.0] * (count1 - 1)
	a.insert(i,1.0)
	onehot_1.append(a)

for i in range(count2):
	a = [0.0] * (count2 - 1)
	a.insert(i,1.0)
	onehot_2.append(a)	

for i in range(count3):
	a = [0.0] * (count3 - 1)
	a.insert(i,1.0)
	onehot_3.append(a)	

#print type1, count1
#print type2, count2
#print type3, count3

for i in range(datasize):	
	n = int(labeldict[traindata[i][featuresize]])
	if (n >= 2):
		tmp = traindata[i]
		for j in range(10**(4-n)):
			traindata.append(tmp)

datasize = len(traindata)
print datasize

random.shuffle(traindata)

#traindata = np.array(traindata)
c = [0, 0, 0, 0, 0]
labeldata = []
for i in range(datasize):
	label = labeldict[traindata[i][featuresize]]
	#if label == 1 and i % 5 != 0:
	#		traindata = np.delete(traindata,i,axis = 0)
	#		continue		
	labeldata.append(label)
	if (i < int(datasize*0.95)):
		c[int(labeldict[traindata[i][featuresize]])] += 1

datasize = len(traindata)

for i in range(datasize):
	if (len(traindata[i]) > featuresize):
		traindata[i].pop()
	for j in range(featuresize):
		traindata[i][j] = float(traindata[i][j])
print c

w = float(np.array(c).sum())/np.array(c)

for i in range(datasize):
	a = onehot_1[int(traindata[i][1])]
	b = onehot_2[int(traindata[i][2])]
	c = onehot_3[int(traindata[i][3])]
	traindata[i] = traindata[i] + a + b + c
	del traindata[i][1]
	del traindata[i][1]
	del traindata[i][1]




datasize = len(traindata)

x_train = np.array(traindata)
y_train = np.array(labeldata)
x_val = np.array(traindata[int(datasize*0.95):])
y_val = np.array(labeldata[int(datasize*0.95):])
'''
param = {}
param['objective'] = 'multi:softmax'
param['max_depth'] = 6
param['n_estimators'] = 100
XGBC = XGBClassifier(objective = 'multi:softmax',max_depth = 6,n_estimators = 200)

XGBC.fit(x_train,y_train)
#evals_result = XGBC.evals_result()

#print evals_result
'''


#######################GBC
# GBC = GradientBoostingClassifier(n_estimators = 200)
# GBC.fit(x_train,y_train)
###################
# model = SVC(kernel="rbf", C=1.0, gamma=1e-4)
# model.fit(x_train,y_train)

num_trees = 10
weight = {}
for i in range(5):
	weight[i] = w[i]
model = RandomForestClassifier(n_estimators=num_trees,class_weight='balanced')
model.fit(x_train,y_train)

val_ans= model.predict(x_val)
acc = accuracy_score(y_val,val_ans)

print acc 
'''

xg_train = xgb.DMatrix( x_train, label=y_train)
xg_val = xgb.DMatrix(x_val ,label=y_val)

print x_train.shape, y_train.shape


# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.05
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 6
param['num_class'] = 5

watchlist = [ (xg_train,'train'), (xg_val, 'test') ]
num_round = 30
bst = xgb.train(param, xg_train, num_round, watchlist );
# get prediction
pred = bst.predict( xg_val );

print ('predicting, classification error=%f' % (sum( int(pred[i]) != y_val[i] for i in range(len(y_val))) / float(len(y_val)) ))

'''

testdata = []
with open('test.in', 'r') as testfile:
	tmp = testfile.readlines()
	for i in range(len(tmp)):
		testdata.append(tmp[i].split('\n')[0].split(','))

print len(testdata)

for i in range(len(testdata)):
	if (testdata[i][1] in type1):
		testdata[i][1] = type1.index(testdata[i][1])
	else:
		testdata[i][1] = 0
	if (testdata[i][2] in type2):
		testdata[i][2] = type2.index(testdata[i][2])
	else:
		testdata[i][2] = 0
	if (testdata[i][3] in type3):
		testdata[i][3] = type3.index(testdata[i][3])
	else:
		testdata[i][3] = 0

for i in range(len(testdata)):
	for j in range(featuresize):
		testdata[i][j] = float(testdata[i][j])

for i in range(len(testdata)):
	a = onehot_1[int(testdata[i][1])]
	b = onehot_2[int(testdata[i][2])]
	c = onehot_3[int(testdata[i][3])]
	testdata[i] = testdata[i] + a + b + c
	del testdata[i][1]
	del testdata[i][1]
	del testdata[i][1]


x_test = np.array(testdata)
#xg_test = xgb.DMatrix(x_test)
###########GBC  
# predict = GBC.predict( x_test );
##################
cc = model.predict(x_test)
prob = model.predict_proba(x_test)
# m = np.mean(prob,axis=0)


with open('answer_rb.csv', 'w') as ansfile:
	ansfile.write('id,label\n')
	for i in range(len(cc)):
		if prob[i][2] > 0:
			ansfile.write(str(i+1) + ',' + '2' + '\n')
		else:
			ansfile.write(str(i+1) + ',' + str(int(cc[i])) + '\n')
