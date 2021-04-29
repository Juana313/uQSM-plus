from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, glob
import numpy as np
import h5py, math
import nibabel as nib
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model 

config = {}
config["input_shape"] = [192, 192, 192]

config['model_file'] = 'model.h5'
config['weight_file'] = 'model_weight.h5'
config['save_predname']  = 'uQSMplus_Pred.nii.gz'

config['data_folder'] = r'D:\Projects\QSM\Data'
config['save_folder'] = r'D:\Projects\QSM\MIDL2021\Data'
config['iFreq_filename'] = 'rdf_resharp4.nii.gz'
config['mask_filename']  = 'brainmask.nii.gz'


config["overwrite"] = True

def load_model_and_weight(model_file, weight_file):
    try:
        # load json and create model
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        
        # load weights into new model
        model.load_weights(weight_file, by_name=True)
        print("Loaded model from disk")    
        
        return model
    except ValueError as error:
        raise error 
    
def saveNifti(dataIn, fileName, affMatrix=None):
    if affMatrix is None:
        affMatrix = np.eye(4)
        
    img = nib.Nifti1Image(dataIn, affMatrix)
    nib.save(img, fileName)  
    
def readNifti(fileName):
    img = nib.load(fileName)
    return img.get_data(), img.affine


def doPrediction(model, dataFolder, overlap=32):
    f = open('prediction.log', 'w')
    
    for root, dirs, files in os.walk(dataFolder):
        for subdir in dirs[:]:
            srcfolder = os.path.join(root, subdir)   #subdir
            print(srcfolder)
            
            if not os.path.exists(os.path.join(os.path.abspath(root), subdir, config['iFreq_filename'])):
                print('skip path %s' % (os.path.join(root, subdir)), file=f)
                f.flush()
                continue
            
            if not config["overwrite"]:
                if os.path.exists(os.path.join(os.path.abspath(root), subdir, config['save_predname'])):
                    print('exist path %s' % (os.path.join(root, subdir)), file=f)
                    f.flush()
                    continue
    
            # ---------------------------------------------------------------------
            # load field map
            iFreq, affine = readNifti(os.path.join(os.path.abspath(root), subdir, config['iFreq_filename']))  
            nxo, nyo, nzo = iFreq.shape
            voxel_size = (affine[0,0], affine[1,1], affine[2,2])
            
            mask, _ = readNifti(os.path.join(os.path.abspath(root), subdir, config['mask_filename']))
            mask = (iFreq!=0)
            iFreq *= mask
            
            
            TE = 25000 * 1e-6;           
            B0 = 3;                      
            gyro = 2*np.pi*42.58;
            
            iFreq /= (TE*B0*gyro)
            
            # ---------------------------------------------------------------------
            # Get the field map ROI
            d1 = np.max(np.max(mask, axis=1), axis=1)
            d1first = np.nonzero(d1)[0][0]
            d1last = np.nonzero(d1)[0][-1]
        
            d2 = np.max(np.max(mask, axis=0), axis=1)
            d2first = np.nonzero(d2)[0][0]
            d2last = np.nonzero(d2)[0][-1]
        
            d3 = np.max(np.max(mask, axis=0), axis=0)
            d3first = np.nonzero(d3)[0][0]
            d3last = np.nonzero(d3)[0][-1]                        
            
            iFreqV = iFreq[d1first:d1last+1, 
                              d2first:d2last+1,
                              d3first:d3last+1]
            maskV = mask[d1first:d1last+1, 
                         d2first:d2last+1,
                         d3first:d3last+1]            
            
            nx, ny, nz = iFreqV.shape
            cnnx, cnny, cnnz = config["input_shape"][0], config["input_shape"][1], config["input_shape"][2]
            
            print("nx = %d, ny = %d, nz = %d" % (nx, ny, nz), file=f)
            print("cnnx = %d, cnny = %d, cnnz = %d" % (cnnx, cnny, cnnz), file=f)
            f.flush()

            
            if nx>cnnx:
                iFreqV = iFreqV[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
                maskV  = maskV[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
            elif nx<cnnx:
                iFreqV = np.pad(iFreqV, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                maskV  = np.pad(maskV, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        
            if ny>cnny:
                iFreqV = iFreqV[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
                maskV  = maskV[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
            elif ny<cnny:
                iFreqV = np.pad(iFreqV, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                maskV  = np.pad(maskV, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        
            if nz>cnnz:
                iFreqV = iFreqV[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
                maskV  = maskV[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
            elif nz<cnnz:
                iFreqV = np.pad(iFreqV, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                maskV  = np.pad(maskV, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                    
                    
            # ---------------------------------------------------------------------
            # Do prediction    
            Pred = model.predict([100*iFreqV[np.newaxis,np.newaxis,:,:,:], maskV[np.newaxis,np.newaxis,:,:,:]])
            Pred = (Pred[0,0])/100*3.0
            
            if nx>cnnx:
                Pred = np.pad(Pred, (((nx-cnnx)//2, (nx-cnnx)-(nx-cnnx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            elif nx<cnnx:
                Pred = Pred[(cnnx-nx)//2:(cnnx-nx)//2+nx,:,:]
        
            if ny>cnny:
                Pred = np.pad(Pred, ((0,0),((ny-cnny)//2, (ny-cnny)-(ny-cnny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            elif ny<cnny:
                Pred = Pred[:,(cnny-ny)//2:(cnny-ny)//2+ny,:]
        
            if nz>cnnz:
                Pred = np.pad(Pred, ((0,0),(0,0),((nz-cnnz)//2, (nz-cnnz)-(nz-cnnz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            elif nz<cnnz:
                Pred = Pred[:,:,(cnnz-nz)//2:(cnnz-nz)//2+nz]

            # ---------------------------------------------------------------------
            Pred3D = np.zeros((nxo, nyo, nzo))
            Pred3D[d1first:d1last+1, d2first:d2last+1, d3first:d3last+1] = Pred
            Pred3D *= mask
          
            # ---------------------------------------------------------------------  
            saveNifti(RDFPred3D*mask, os.path.join(config['save_folder'], subdir, config['save_predname']), affine)            

def main():    
    model = load_model_and_weight(config['model_file'], config['weight_file']) 
    print(model.summary())    
    
    doPrediction(model, config['data_folder'])

if __name__ == "__main__":
    main()
