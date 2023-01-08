import numpy as np
from PIL import Image
import random 
import os
#import pickle
from sklearn.model_selection import train_test_split
#!pip3 install imageio
import imageio
from keras.utils import to_categorical

training_inputs_orig = []
y_orig = []

def create_training_data(fpath, class_num):
  img = imageio.imread(fpath)
  training_inputs_orig.append(img)
  y_orig.append(class_num)

create_training_data("./images/Aria/Aria_1.png", 1)

shuflist = []
for i in range(len(training_inputs_orig)):
  shuflist.append([training_inputs_orig[i], y_orig[i]])

from random import shuffle
shuffle(shuflist)

training_inputs = []
y = []
for j in range(len(shuflist)):
  ts = shuflist[j]
  training_inputs.append(ts[0])
  y.append(ts[1])

training_inputs = np.array(training_inputs)
y = np.array(y)
print (training_inputs.shape, y.shape)

# split the new DataFrame into training and testing sets
x_train, x_test, y_orig_train, y_orig_test = train_test_split(training_inputs, y, random_state=1)

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(-1,28,28,1)
#print (y_orig_train.shape)
#y_train = np.array(to_categorical(y_orig_train.astype('float32')))
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_orig_train-1, num_classes=10)
#print (y_train.shape)

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(-1,28,28,1)
#y_test = np.array(to_categorical(y_orig_test.astype('float32')))
y_test = np_utils.to_categorical(y_orig_test-1, num_classes=10)

print ("x_train:", x_train.shape, "y_train:", y_train.shape, "x_test:", x_test.shape, "y_test:", y_test.shape)

# ~~~~~~~~~~~~~~~~~~~~~~ model start ~~~~~~~~~~~~~~~~~~~~~~~~

import keras
from keras.models import Model
from keras.layers import Conv2D, Dense, Input, Reshape, Lambda, Layer, Flatten
#from keras import backend as K
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import initializers
from keras.layers.core import Activation

input_shape = Input(shape=(28,28,1))  # size of input image is 28*28

# a convolution layer output shape = 20*20*256
conv1 = Conv2D(256, (9,9), activation = 'relu', padding = 'valid')(input_shape)

# convolution layer with stride 2 and 256 filters of size 9*9
conv2 = Conv2D(256, (9,9), strides = 2, padding = 'valid')(conv1)

# reshape into 1152 capsules of 8 dimensional vectors
reshaped = Reshape((6*6*32,8))(conv2)

def squash(inputs):
    # take norm of input vectors
    squared_norm = K.sum(K.square(inputs), axis = None, keepdims = True)

    # use the formula for non-linear function to return squashed output
    return ((squared_norm/(1+squared_norm))/(K.sqrt(squared_norm+K.epsilon())))*inputs

# squash the reshaped output to make length of vector b/w 0 and 1
squashed_output = Lambda(squash)(reshaped)

class DigitCapsuleLayer(Layer):
    # creating a layer class in keras
    def __init__(self, **kwargs):
        super(DigitCapsuleLayer, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get('glorot_uniform')

    def build(self, input_shape):
        # initialize weight matrix for each capsule in lower layer
        self.W = self.add_weight(shape = [10, 6*6*32, 16, 8], initializer = self.kernel_initializer, name = 'weights')
        # g self.W = self.add_weight(shape = [3, 6*6*32, 1, 8], initializer = self.kernel_initializer, name = 'weights')
        self.built = True

    def call(self, inputs):
        inputs = K.expand_dims(inputs, 1)
        inputs = K.tile(inputs, [1, 10, 1, 1])
        # g inputs = K.tile(inputs, [1, 3, 1, 1])
        # matrix multiplication b/w previous layer output and weight matrix
        inputs = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs)
        b = tf.zeros(shape = [K.shape(inputs)[0], 10, 6*6*32])
        # g b = tf.zeros(shape = [K.shape(inputs)[0], 3, 6*6*32])

# routing algorithm with updating coupling coefficient c, using scalar product b/w input capsule and output capsule
        for i in range(3-1):
            c = tf.nn.softmax(b, dim=1)
            s = K.batch_dot(c, inputs, [2, 2])
            v = squash(s)
            b = b + K.batch_dot(v, inputs, [2,3])

        return v
    def compute_output_shape(self, input_shape):
      return tuple([None, 10, 16])
      # g return tuple([None, 3, 1])


def output_layer(inputs):
    return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

digit_caps = DigitCapsuleLayer()(squashed_output)
outputs = Lambda(output_layer)(digit_caps)

def mask(outputs):

    if type(outputs) != list:  # mask at test time
        norm_outputs = K.sqrt(K.sum(K.square(outputs), -1) + K.epsilon())
        # g y  = K.one_hot(indices=K.argmax(norm_outputs, 1), num_classes = 3)
        # g y = Reshape((3,1))(y)
        y  = K.one_hot(indices=K.argmax(norm_outputs, 1), num_classes = 10)
        y = Reshape((10,1))(y)
        return Flatten()(y*outputs)

    else:    # mask at train time
        # g y = Reshape((3,1))(outputs[1])
        y = Reshape((10,1))(outputs[1])
        masked_output = y*outputs[0]
        return Flatten()(masked_output)

# g inputs = Input(shape = (3,))
inputs = Input(shape = (10,))
masked = Lambda(mask)([digit_caps, inputs])
masked_for_test = Lambda(mask)(digit_caps)

# g decoded_inputs = Input(shape = (1*3,))
decoded_inputs = Input(shape = (16*10,))
dense1 = Dense(512, activation = 'relu')(decoded_inputs)
# g dense2 = Dense(9, activation = 'relu')(dense1)
dense2 = Dense(1024, activation = 'relu')(dense1)
decoded_outputs = Dense(784, activation = 'sigmoid')(dense2)
decoded_outputs = Reshape((28,28,1))(decoded_outputs)

def loss_fn(y_true, y_pred):

    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

decoder = Model(decoded_inputs, decoded_outputs)
model = Model([input_shape,inputs],[outputs,decoder(masked)])
test_model = Model(input_shape,[outputs,decoder(masked_for_test)])

# ~~~~~~~~~~~~~~~~~~~~~~ model end ~~~~~~~~~~~~~~~~~~~~~~~~

m = 128
epochs = 2
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss=[loss_fn,'mse'],loss_weights = [1. ,0.0005],metrics=['accuracy'])
history = model.fit([x_train, y_train],[y_train, x_train], batch_size = m, epochs = epochs, validation_data = ([x_test, y_test],[y_test,x_test]))

label_predicted, image_predicted = model.predict([x_test, y_test])

n_samples = 10
sample_predicted = []
sample_test = []

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    sample_test.append(y_orig_test[index])
    sample_image = x_test[index].reshape(28, 28)
    plt.imshow(sample_image, cmap="binary")
    plt.title("Label:" + str(y_orig_test[index]))
    plt.axis("off")

plt.show()

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    sample_image = image_predicted[index].reshape(28, 28)
    sample_predicted.append(np.argmax(label_predicted[index]))
    plt.imshow(sample_image, cmap="binary")
    plt.title("Predicted:" + str(np.argmax(label_predicted[index])))
    plt.axis("off")

plt.show()

#plots
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['model_1_acc']
val_acc = history.history['val_model_1_acc']
plt.plot(epochs, acc, color='red', label='Training accuracy')
plt.plot(epochs, val_acc, color='green', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


from sklearn.metrics import confusion_matrix
# Confusion matrix
print (np.array(sample_test).shape)
print (np.array(sample_predicted).shape)
samp_test = np.array(sample_test).flatten()
sample_test = np.array(sample_test).round()
samp_pred = np.array(sample_predicted).flatten()

from sklearn.metrics import accuracy_score
accuracy_sc = accuracy_score(sample_test, samp_pred)
print("accuracy score: {0:.2f}%".format(accuracy_sc*100))

cm = confusion_matrix(samp_test, samp_pred)


# Plot normalized confusion matrix
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
plt.figure(figsize=(10,10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
classNames = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=90)
plt.yticks(tick_marks, classNames)
#for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#    plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
