import pickle
import os, glob
from random import shuffle
import nibabel as nib
import numpy as np
import h5py
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras import optimizers
from tensorflow.keras.utils import multi_gpu_model


def saveH5(dataIn, fileName):
    if type(dataIn) is dict:
        # write to h5file
        f = h5py.File(fileName, "w")
        for key in dataIn.keys():
            if dataIn[key] is not None:
                dset = f.create_dataset(key, data=dataIn[key]) 
        f.close()
        return 0
    else:
        return -1

def readH5(fileName):
    f = h5py.File(fileName, "r")
    dataOut = {}
    
    keys = [key for key in f.keys()]
    for key in keys:  
        dataOut[key] = f[key][:]
    f.close()    
    return dataOut
    
def saveNifti(dataIn, fileName, affMatrix=None):
    if affMatrix is None:
        affMatrix = np.eye(4)
        
    img = nib.Nifti1Image(dataIn, affMatrix)
    nib.save(img, fileName)  
    
def readNifti(fileName):
    img = nib.load(fileName)
    return img.get_fdata()

# ====================================================================
def save_model(model, model_file):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)
        
def load_model_json(model_file):
    print("Loading pre-trained model")
    try:
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)        
        return loaded_model
    except ValueError as error:
        raise error    

def save_weight(model, weight_file):
    model.save_weights(weight_file)

    
def load_model_and_weight(model_file, weight_file):
    try:
        # load json and create model
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        #model = model_from_json(loaded_model_json, {'GConv3D':GConv3D})
        model = model_from_json(loaded_model_json)

        # load weights into new model
        model.load_weights(weight_file, by_name=True)
        print("Loaded model from disk")    
        
        return model
    except ValueError as error:
        raise error             


# ====================================================================
def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

def fetch_data_files(datasets_path, modalities, subdir_filter="*"):
    training_data_files = list()

    for subject_dir in glob.glob(os.path.join(datasets_path, subdir_filter)):
        subject_files = list()
        files_existing = True

        for modality in modalities:
            if os.path.exists(os.path.join(subject_dir, modality)):
                subject_files.append(os.path.join(subject_dir, modality))
            else:
                files_existing = False

        if files_existing:
            training_data_files.append(subject_files)
    return training_data_files


def get_validation_split(data_files, training_file, validation_file, data_split=0.8, overwrite=False):
    def split_list(input_list, split=0.8, shuffle_list=True):
        if shuffle_list:
            shuffle(input_list)
        n_training = int(len(input_list) * split)
        training = input_list[:n_training]
        testing = input_list[n_training:]
        return training, testing
    
    if overwrite or not os.path.exists(training_file) or not os.path.exists(validation_file):
        print("Creating validation split...")
        nb_samples = len(data_files)
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split,shuffle_list=True)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)

