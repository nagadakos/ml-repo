import numpy as np
import keras
from keras.models import  Sequential
from keras.layers import  Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import  Conv2D, MaxPooling2D
from keras.utils import  np_utils
from keras.datasets import  mnist
from keras import  backend as K
from matplotlib import pyplot as plt

# seed init
np.random.seed(123)  # for reproducibility


batch_size = 64
num_classes = 10
epochs = 100
# input image dimensions
img_rows, img_cols = 28, 28
# Load pre-shuffled MNIST data into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(keras.__version__)
print( x_train.shape)
# (60000, 28, 28)

# Backend is Tensorflow which accepts input as Batch, Number Channels, height, width
# So we have to reshape the input to that format.
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(input_shape)

# Model Definition
# 2 conv2d layers, a max pool and a a Dropout. Dropout layers randomly
# dropping outputs of a layer, they help combat overfitting.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
activation='relu',
input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.Adam(),
metrics=['accuracy'])
history = model.fit(x_train, y_train,
batch_size=batch_size,
epochs=epochs,
verbose=1,
validation_data=(x_test, y_test))

# report accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('Target loss:', score[0])
print('Taerget accuracy:', score[1])

# save the model 
model_json = model.to_json()
with open("model_no_drop.json", "w") as json_file:
    json_file.write(model_json)
    model.save_weights("model_no_drop.h5")
    for keys in history.history.keys():
        print(keys)
with open("mnistKeras_no_drop_report.txt", "a") as f:
    for i in range(len(history.history['acc'])):
        print(i)
        f.write("{0:.4f} {1:.4f} {2:.4f} {3:.4f}\n".format(history.history['acc'][i], history.history['loss'][i], history.history['val_acc'][i], history.history['val_loss'][i]))
