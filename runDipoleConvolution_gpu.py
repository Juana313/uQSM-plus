from __future__ import print_function
import numpy as np
import os, glob, shutil
from time import gmtime, strftime
import nibabel as nib
from tensorflow.keras import  Model
from tensorflow.keras.layers import Input
from fmLayer import CalFMLayer

config = {}

config['dataFolder'] = r'.\fb_roi_96x96x96'
config['matrix_size'] = [96, 96, 96]
config['pad_size'] = [16, 16, 16]
config['voxel_size'] = [1, 1, 1.0]


def saveNifti(dataIn, fileName, affMatrix=None):
    if affMatrix is None:
        affMatrix = np.eye(4)
    
    img = nib.Nifti1Image(dataIn, affMatrix)
    nib.save(img, fileName)

def readNifti(fileName):
    if not os.path.exists(fileName):
        return None, None
    img = nib.load(fileName)
    return img.get_data(), img.affine

def main(dataFolder):
    count = 0
    f = open('out.log', 'w')   
    
    # --------------------------------------------------
    for root, dirs, files in os.walk(dataFolder):
        for subdir in dirs[:]:
            srcfolder = os.path.join(root, subdir)   #subdir
            print("data - %s" % (srcfolder), file=f)
            f.flush()
                                    
            if os.path.exists(os.path.join(os.path.abspath(dataFolder), subdir, 'fmap_suscp.nii.gz')):
                print("subdir %s - skip" % (subdir), file=f)
                f.flush()
                continue 
            
            Nx = config['matrix_size'][0] + 2*config['pad_size'][0]
            Ny = config['matrix_size'][1] + 2*config['pad_size'][1]
            Nz = config['matrix_size'][2] + 2*config['pad_size'][2]
            
            voxel_size = config['voxel_size']
            FOV = [Nx*voxel_size[0], Ny*voxel_size[1], Nz*voxel_size[2]]
            kx_squared = np.fft.ifftshift(np.arange(-Nx/2.0, Nx/2.0)/float(FOV[0]))**2
            ky_squared = np.fft.ifftshift(np.arange(-Ny/2.0, Ny/2.0)/float(FOV[1]))**2
            kz_squared = np.fft.ifftshift(np.arange(-Nz/2.0, Nz/2.0)/float(FOV[2]))**2

            [ky2_3D,kx2_3D,kz2_3D] = np.meshgrid(ky_squared,kx_squared,kz_squared)
            kernel = 1/3.0 - kz2_3D/(kx2_3D + ky2_3D + kz2_3D)
            kernel[0,0,0] = 0
            

            def fm_model(input_shape=(1,320,320,320)):
                suscp = Input((1, input_shape[1], input_shape[2], input_shape[3]))
                qsm_kernel = Input((1, input_shape[1], input_shape[2], input_shape[3]))
                
                fm = CalFMLayer()([suscp, qsm_kernel])
                 
                model = Model(inputs=[suscp, qsm_kernel],
                              outputs=[fm])

                return model     
           
            model = fm_model(input_shape=(1,Nx, Ny, Nz))

            if not os.path.exists(os.path.join(os.path.abspath(dataFolder), subdir, 'suscp.nii.gz')):
                print("subdir %s - suscp.nii.gz not exit skip" % (subdir), file=f)
                f.flush()
                continue

            try:
                suscpData3D, aff = readNifti(os.path.join(os.path.abspath(dataFolder), subdir, 'suscp.nii.gz'))
                maskData = (suscpData3D!=0)
                
                pad_nx, pad_ny, pad_nz = config['pad_size'][0], config['pad_size'][1], config['pad_size'][2]
                suscpData3D_pad = np.pad(suscpData3D, ((pad_nx, pad_nx),(pad_ny, pad_ny),(pad_nz, pad_nz)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                maskData_pad = np.pad(maskData, ((pad_nx, pad_nx),(pad_ny, pad_ny),(pad_nz, pad_nz)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                
                # -------------------------------------------
                fMap = model.predict([suscpData3D_pad[np.newaxis,np.newaxis,:,:,:], kernel[np.newaxis,np.newaxis,:,:,:]])            
                fMap = fMap[0,0,pad_nx:-pad_nx, pad_ny:-pad_ny, pad_nz:-pad_nz]

                saveNifti(fMap, os.path.join(os.path.abspath(dataFolder), subdir, 'fmap.nii.gz'), aff)

                print("subdir %s - done" % (subdir), file=f)
                f.flush()   
            except:
                continue
            
            # -------------------------------------------
            count += 1
            f.flush()
            
    f.close()
if __name__ == "__main__":
    main(os.path.abspath(config['dataFolder']))

