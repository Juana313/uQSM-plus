
import os,math
import numpy as np
import nibabel as nib
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, Callback

from unet import unet_model_3d, semi_model_t
from datagenerator import DataGenerator
from utils import pickle_dump, pickle_load, fetch_data_files, get_validation_split, save_model
from callbacks import get_callbacks
from loss import tv_loss

config = dict()
config["pool_size"] = (2,2,2)           # pool size for the max pooling operations
config["n_base_filters"] = 24           # num of base kernels
config["conv_kernel"] = (3,3,3)         # convolutional kernel shape
config["layer_depth"] = 5               # unet depth
config["deconvolution"] = False         # if False, will use upsampling instead of deconvolution
config["batch_normalization"] = False    # Using batch norm
config["activation"] = "linear"


config["initial_learning_rate"] = 0.0002
config["batch_size"] = 2
config["patch_size"] = [96, 96, 96]
config["voxel_size"] = [1., 1., 1.]

config["valid_batch_size"] = 1
config["valid_patch_size"] = [160, 160, 160]

config["model_file"] = os.path.abspath("model.h5")                  # save the model structure and weight seperately
config["model_weight_file"] = os.path.abspath("model_weight.h5")


def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    # training data
    data_files = fetch_data_files(r'D:\Projects\QSM\Data', ['rdf_resharp4.nii.gz', 'brainmask.nii.gz', 'mag.nii.gz'])

    print("num of datasets %d" % (len(data_files)))

    # -------------------------------
    # create data generator for training and validatation
    training_list, validation_list = get_validation_split(data_files,
                                                          'training.pkl',
                                                          'val.pkl',
                                                          data_split=1,
                                                          overwrite=True)
    training_generator = DataGenerator(data_files,
                                       training_list,
                                       batch_size = config["batch_size"],
                                       patch_size = config["patch_size"],
                                       voxel_size = config["voxel_size"],
                                       shuffle=True)
    validation_generator = DataGenerator(data_files,
                                         validation_list,
                                         batch_size = config["valid_batch_size"],
                                         patch_size = config["valid_patch_size"],
                                         voxel_size = config["voxel_size"],
                                         shuffle=False)
    
    # -------------------------------
    # create the model
    umodel = unet_model_3d(pool_size=config["pool_size"],
                           deconvolution=config["deconvolution"],
                           depth=config["layer_depth"] ,
                           n_outputs=1,
                           n_base_filters=config["n_base_filters"],
                           kernel = config["conv_kernel"],
                           batch_normalization=config["batch_normalization"],
                           activation_name=config["activation"]) 
    
            
            
    print("model summary")
    print(umodel.summary())
    save_model(umodel, config["model_file"])
    
    model = semi_model_t(umodel)
        
    optimizer = optimizers.Adam(lr=config["initial_learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    alpha = K.variable(1.0)
    beta = K.variable(0.06)
    gamma_roi, gamma_out = K.variable(0.0), K.variable(0.0)
    model.compile(loss=['mse', tv_loss, 'mae', 'mae'],
                  loss_weights=[alpha, beta, gamma_roi, gamma_out],
                  optimizer=optimizer)    
    
    callbacks=get_callbacks(model = [model, umodel], model_weights = [alpha, beta, gamma_roi, gamma_out],
                                                    weight_file = 'model_weight.h5',
                                                    initial_learning_rate=config["initial_learning_rate"],
                                                    learning_rate_drop=0.5, 
                                                    learning_rate_epochs=None, 
                                                    learning_rate_patience=20, 
                                                    early_stopping_patience=None)
    
    model.fit_generator(    generator=training_generator,
                            steps_per_epoch=200,
                            epochs=15,
                            max_queue_size=4,
                            use_multiprocessing=False,
                            workers=2,
                            validation_data=None, #validation_generator,
                            validation_steps=0, #len(validation_list)//config["batch_size"],
                            callbacks = callbacks
                            )

    umodel.save_weights('model_weight.h5', overwrite=True)
 
if __name__ == "__main__":
    main()
