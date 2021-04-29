import math
import numpy as np
from functools import partial

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, Callback, LambdaCallback


# ------------------------------------------------------
class LearningRateDecay(Callback):
    def __init__(self, model, model_weights=None, initial_lrate=0.001, decay_rate=0.1, n_batch=100):
        super(LearningRateDecay, self).__init__()
        self.n_steps        = 0
        self.initial_lrate  = initial_lrate
        self.modelx         = model
        self.decay_rate     = np.float32(decay_rate)
        self.n_batch        = n_batch
        self.model_weights = model_weights
    
    def on_train_begin(self, logs={}):
        pass
    
    def on_epoch_begin(self, epoch, logs={}):
        pass
    
    def on_batch_begin(self, batch, logs={}):
        pass
    
    def on_batch_end(self, batch, logs={}):
        self.n_steps += 1
        if self.n_steps%self.n_batch == 0:
            x = self.n_steps//self.n_batch
            lrate = self.initial_lrate * math.exp(-self.decay_rate*x)
            K.set_value(self.model.optimizer.lr, lrate)
            print('lr:', K.eval(self.model.optimizer.lr))
            

    def on_epoch_end(self, epoch, logs={}):
        w1 = K.get_value(self.model_weights[0])
        w2 = K.get_value(self.model_weights[1])
        w3_roi = K.get_value(self.model_weights[2])
        w4_out = K.get_value(self.model_weights[3])
        
        print('model weights:', w1, w2, w3_roi, w4_out)
        
        w = 1.5
        if epoch < 6:
            w3_roi = w*(epoch+1)/6
            w4_out = w*(epoch+1)/6
        elif epoch < 12:
            w3_roi = w - w*(epoch-6)/6
            w4_out = w - w*(epoch-6)/6
        else:
            w3_roi = 0.1
            w4_out = 0.1
            
        
        K.set_value(self.model_weights[2], w3_roi)
        K.set_value(self.model_weights[3], w4_out)
        
        print('model weights:', w1, w2, w3_roi, w4_out)


        
    def on_train_end(self, logs={}):
        pass

# ------------------------------------------------------
class LearningRatePrinter(Callback):
    def __init__(self, model):
        super(LearningRatePrinter, self).__init__()
        #self.model = model
    def on_epoch_begin(self, epoch, logs={}):
        print('lr:', K.eval(self.model.optimizer.lr))

# ------------------------------------------------------
class CustomModelCheckpoint(Callback):
    def __init__(self, model, path):
        super(CustomModelCheckpoint, self).__init__()

        self.path = path
        self.best_loss = np.inf
        self.model_for_saving = model    # We set the model (non multi gpu) under an other name

    def on_epoch_end(self, epoch, logs=None):
        #loss = logs['val_loss']
        self.model_for_saving.save_weights('model_weight_%d.h5'%(epoch), overwrite=True)
        self.model_for_saving.save_weights(self.path, overwrite=True)
        #if loss<self.best_loss:
        #    self.model_for_saving.save_weights(self.path, overwrite=True)
        #    self.best_loss = loss

# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def get_callbacks(model, model_weights, weight_file,initial_learning_rate=0.0001, 
                  learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", 
                  verbosity=1, early_stopping_patience=None):
    callbacks = list()
    callbacks.append(CustomModelCheckpoint(model[1], weight_file))
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, 
                                                       initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, 
                                                       epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, 
                                           patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    callbacks.append(LearningRatePrinter(model[0]))
    callbacks.append(LearningRateDecay(model, model_weights=model_weights, initial_lrate=initial_learning_rate, decay_rate=0.1, n_batch=200))
    return callbacks