import logging as log

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Reshape, Flatten

def build_model(is_train):  
    n_inputs = 1200
    in_shape = (20, 20, 3)
    kernel = (3, 3)

    model = Sequential()

    # reshape vector to ([x, y, z],...)
    model.add(Reshape(in_shape, input_shape=(n_inputs,)))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model
