import keras
import numpy as np
import pickle as pickle
import sys
import random
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Dropout,BatchNormalization,advanced_activations,UpSampling2D,Input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

import tensorflow as tf
import sklearn.utils as ut
tf.python.control_flow_ops = tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config = config)
path = sys.argv[1]
modelname = sys.argv[2] 
from keras import backend as K
Threshold = 0.999
X_train = np.array(pickle.load(open(path + "all_label.p",'r'))).reshape(5000,3,32,32)
Y_train = []
for i in xrange(0,5000):
	Y_train.append(i/500)
Y_train = np_utils.to_categorical(Y_train, nb_classes=10)
list1_shuf = []
list2_shuf = []
index_shuf = range(len(X_train))
random.shuffle(index_shuf)
for i in index_shuf:
	list1_shuf.append(X_train[i])
	list2_shuf.append(Y_train[i])
X_train = np.array(list1_shuf)
Y_train = np.array(list2_shuf)



Val_X = X_train[4500:len(X_train)-1]
Val_Y = Y_train[4500:len(Y_train)-1]
X_train = X_train[0:4500]
Y_train = Y_train[0:4500]

X_test = pickle.load(open(path + "test.p",'r'))
X_test = np.array(X_test['data'])
X_test = X_test.reshape(10000,3072)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
Val_X = Val_X.astype('float32')
X_train /= 255
X_test /= 255
Val_X /= 255
print "import unlabeled data"
unlableData = np.array(pickle.load(open(path + "all_unlabel.p",'r'))).reshape(45000,3,32,32)
unlableData = unlableData.astype('float32')
unlableData /= 255

en = unlableData.reshape(unlableData.shape[0],3072)
X_flat = X_train.reshape(X_train.shape[0],3072)
inputs = Input(shape=(3072,))
x = Dense(3072, activation='relu')(inputs)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
encoder = Dense(256, activation='relu')(x)
x = Dense(512, activation='relu')(encoder)
x = Dense(1024, activation='relu')(x)
decoder = Dense(3072, activation='sigmoid')(x)
autoencoder = Model(input=inputs, output=decoder)
#model.compile(loss='binary_crossentropy', optimizer='adad')
'''
autoencoder = Sequential()
autoencoder.add(Dense(3072,input_dim = 3072))
autoencoder.add(Dense(1024,activation='relu'))
autoencoder.add(Dense(512,activation='relu'))
autoencoder.add(Dense(256,activation='relu'))
autoencoder.add(Dense(512,activation='relu'))
autoencoder.add(Dense(1024,activation='relu'))
autoencoder.add(Dense(3072,activation ='sigmoid'))
'''
autoencoder.summary()
autoencoder.compile(optimizer = 'adadelta',loss = 'categorical_crossentropy',metrics = ['accuracy'])
autoencoder.fit(en,en,nb_epoch=60, batch_size=32)

encoded = Model(input = inputs,output = encoder)

hidden_outputs =  encoded.predict(X_flat)
#get_layer_output = K.function([autoencoder.layers[0].input],
#                             [autoencoder.layers[3].output])
#hidden_outputs = get_layer_outpu
print hidden_outputs
hidden_outputs = hidden_outputs.reshape(hidden_outputs.shape[0],1,16,16)
model = Sequential()
model.add(Convolution2D(
	nb_filter = 64,
	nb_row = 3,
	nb_col = 3,
	border_mode = 'same',
    input_shape =(1,16,16))
)
model.add(keras.layers.advanced_activations.ELU(alpha=1))
model.add(Convolution2D(64,3,3,border_mode = 'same'))
#model.add(advanced_aci('elu'))
model.add(keras.layers.advanced_activations.ELU(alpha=1))

model.add(BatchNormalization(axis = 1))
model.add(MaxPooling2D(pool_size = (3,3),border_mode = 'same'))
model.add(Dropout(0.3))
model.add(Convolution2D(128,3,3,border_mode = 'same'))
model.add(keras.layers.advanced_activations.ELU(alpha=1))

#model.add(Activation('elu'))
model.add(Convolution2D(128,3,3,border_mode = 'same'))
model.add(keras.layers.advanced_activations.ELU(alpha=1))

#model.add(Activation('elu'))
model.add(BatchNormalization(axis = 1))
model.add(MaxPooling2D(pool_size = (3,3),border_mode = 'same'))
model.add(Dropout(0.3))
model.add(Convolution2D(256,3,3,border_mode = 'same'))
model.add(keras.layers.advanced_activations.ELU(alpha=1))

#model.add(Activation('elu'))
model.add(Convolution2D(256,3,3,border_mode = 'same'))
model.add(keras.layers.advanced_activations.ELU(alpha=1))

#model.add(Activation('elu'))
model.add(BatchNormalization(axis = 1))
model.add(MaxPooling2D(pool_size = (3,3),border_mode = 'same'))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512))
model.add(keras.layers.advanced_activations.ELU(alpha=1))
model.add(BatchNormalization(axis = 1))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr = 1e-4)
model.compile(optimizer = 'adadelta',loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.fit(hidden_outputs,Y_train,nb_epoch = 60,batch_size = 32)
model.save(modelname)

