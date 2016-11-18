import keras
import numpy as np
import pickle as pickle
import random
import sys
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model

from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Dropout,BatchNormalization,advanced_activations
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import sklearn.utils as ut
tf.python.control_flow_ops = tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config = config)
from keras import backend as K
path = sys.argv[1]
modelname = sys.argv[2]
predictionname = sys.argv[3]

X_test = pickle.load(open(path + "test.p",'r'))
X_test = np.array(X_test['data'])
X_test = X_test.reshape(10000,3072)
X_test = X_test.astype('float32')

X_test /= 255
pretrain = load_model("pretrain_model.h5")
model = load_model(modelname)

encoded = pretrain.predict(X_test,batch_size = 50)
encoded = encoded.reshape(encoded.shape[0],1,16,16)
out = model.predict_classes(encoded, batch_size=50, verbose=1)
print out

f = open(predictionname , 'w')
f.write("ID,class\n")
for i in xrange(0,len(out)) :
    tmp = str(i) + "," + str(int(out[i])) + "\n"  
    f.write(tmp)

