import time
import os
import logging as log

from keras.callbacks import EarlyStopping, TensorBoard

def train_model(model, dataset):
    log.info("training model (train on %d samples, validate on %d) ..." % ( \
            len(dataset.Y_train), 
            len(dataset.Y_val) ) )
    
    loss      = 'binary_crossentropy'
    optimizer = 'adam'
    metrics   = ['accuracy']
    
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    earlyStop = EarlyStopping(monitor = 'val_acc', min_delta=0.0001, patience = 5, mode = 'auto')

    log_dir = os.path.join(dataset.path, "logs/{}".format(time.time()))
    tensorboard = TensorBoard( \
            log_dir        = log_dir, 
            histogram_freq = 1, 
            write_graph    = True, 
            write_grads    = True,
            write_images   = True)

    tensorboard.set_model(model)

    return model.fit( dataset.X_train, dataset.Y_train,
            batch_size = 64,
            epochs = 50,
            verbose = 2,
            validation_data = (dataset.X_val, dataset.Y_val),
            callbacks = [tensorboard, earlyStop])
