from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Reshape, Flatten

def build_model(is_train):  
    inputshape = (1200,)
    imgshape   = (20, 20, 3)
    kernel     = (3, 3)
    pooling    = (2, 2)

    model = Sequential()
    # our input vector is flat, but Conv2D wants a 3d
    # shaped tensor (width, height, depth), reshape it.
    model.add( Reshape(imgshape, input_shape=inputshape) )
    # add some convolutional filters
    model.add( Conv2D(32, kernel_size=kernel, activation='relu') )
    model.add( Conv2D(64, kernel, activation='relu') )
    # downsample from the convolutional filters
    model.add( MaxPooling2D(pool_size=pooling) )
    # add some dropout to avoid overfitting
    model.add( Dropout(0.25) )
    # flatten results for the next dense layer
    model.add( Flatten() )
    # this layer is gonna learn how to classify planes
    # according to the inputs that the convolutional 
    # and dropout layers activate.
    model.add( Dense(128, activation='relu') )
    # avoid overfitting
    model.add( Dropout(0.5) )
    # the output layer
    model.add( Dense(2, activation='softmax') )

    return model
