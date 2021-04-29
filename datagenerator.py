import numpy as np
import os
from tensorflow.keras.utils import Sequence
from random import shuffle
import nibabel as nib
from glob import glob
    
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, data_files, training_list, patch_size=[64, 64, 64], voxel_size=[1.,1.,1.], batch_size=2):
        'Initialization'
        self.data_files = data_files
        self.indexes    = training_list
        
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.voxel_size= voxel_size
        
            
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = np.random.choice(self.indexes, size=1)

        # Generate data
        # X1 - measured local field
        # m1 - mask 
        # w1 - weighting matrix
        # k  - dipole kernel
        # X2 - measured local field + perturbed local field
        # y2 - perturbed susceptibility source
        # mroi - mask of perturbed susceptibility source
        # mxroi - mask - mask of perturbed susceptibility source
        X1,m1,w1,k, X2,y2, mroi, mxroi = self.__data_generation(indexes)

        return [X1,m1,w1,k, X2, mroi, mxroi], [np.zeros(k.shape), np.zeros(k.shape), y2, np.zeros(k.shape)]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x1_list = list()
        w1_list = list()
        m1_list = list()
        k_list = list()
        
        x2_list = list()
        y2_list = list()
        mroi_list = list()
        mxroi_list = list()
        
        # Generate data
        for i, index in enumerate(indexes):
            image_list = list()     
            for k, image_file in enumerate(self.data_files[index]):
                image = nib.load(os.path.abspath(image_file))
                image_list.append(image)
               
            subject_data = [image.get_data() for image in image_list]

            rdf = np.asarray(subject_data[0]) # local field
            m = np.asarray(subject_data[1])   # mask
            w = np.asarray(subject_data[2])   # weighting matrix
            
            w *= m

            TE, B0 = 25000 * 1e-6, 3                                
            gyro = 2*np.pi*42.58
            rdf /= (TE*B0*gyro)
            w /= w[m>0].max()
            
            # crop to bounding box of brain
            d1 = np.max(np.max(m, axis=1), axis=1)
            d1first = np.nonzero(d1)[0][0]
            d1last = np.nonzero(d1)[0][-1]
        
            d2 = np.max(np.max(m, axis=0), axis=1)
            d2first = np.nonzero(d2)[0][0]
            d2last = np.nonzero(d2)[0][-1]
        
            d3 = np.max(np.max(m, axis=0), axis=0)
            d3first = np.nonzero(d3)[0][0]
            d3last = np.nonzero(d3)[0][-1]                        
            
            rdf = rdf[d1first:d1last+1, d2first:d2last+1, d3first:d3last+1]
            m = m[d1first:d1last+1, d2first:d2last+1, d3first:d3last+1]
            w = w[d1first:d1last+1, d2first:d2last+1, d3first:d3last+1]
            
            
            #dipole kernel
            voxel_size = self.voxel_size
            cnnx, cnny, cnnz = self.patch_size[0], self.patch_size[1], self.patch_size[2]
            FOV = [cnnx*voxel_size[0], cnny*voxel_size[1], cnnz*voxel_size[2]]
            kx_squared = np.fft.ifftshift(np.arange(-cnnx/2.0, cnnx/2.0)/float(FOV[0]))**2
            ky_squared = np.fft.ifftshift(np.arange(-cnny/2.0, cnny/2.0)/float(FOV[1]))**2
            kz_squared = np.fft.ifftshift(np.arange(-cnnz/2.0, cnnz/2.0)/float(FOV[2]))**2

            [ky2_3D,kx2_3D,kz2_3D] = np.meshgrid(ky_squared,kx_squared,kz_squared)
            kernel = 3*(1/3.0 - kz2_3D/(kx2_3D + ky2_3D + kz2_3D + 1e-18))
            kernel[0,0,0] = 0
            kernel = kernel[np.newaxis,:]
            
            
            for ii in range(self.batch_size):
                # crop image patches
                while 1:
                    ix, iy, iz = 0, 0, 0
                    if rdf.shape[0]-cnnx > 0:
                        ix = np.random.random_integers(rdf.shape[0]-cnnx, size=1)[0] - 1
                    if rdf.shape[1]-cnny > 0:
                        iy = np.random.random_integers(rdf.shape[1]-cnny, size=1)[0] - 1
                    if rdf.shape[2]-cnnz > 0:
                        iz = np.random.random_integers(rdf.shape[2]-cnnz, size=1)[0] - 1
                    
                    rdfPatch = rdf[ix:ix+cnnx, iy:iy+cnny, iz:iz+cnnz]
                    maskPatch = m[ix:ix+cnnx, iy:iy+cnny, iz:iz+cnnz]
                    wPatch = w[ix:ix+cnnx, iy:iy+cnny, iz:iz+cnnz]

                    if maskPatch.sum() > 0.5*cnnx*cnny*cnnz:
                        break
                
                nx, ny, nz = rdfPatch.shape
                
                # padding the patches if needed
                if nx>cnnx:
                    rdfPatch = rdfPatch[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
                    maskPatch  = maskPatch[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
                    wPatch  = wPatch[(nx-cnnx)//2:(nx-cnnx)//2+cnnx,:,:]
                elif nx<cnnx:
                    rdfPatch = np.pad(rdfPatch, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                    maskPatch  = np.pad(maskPatch, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                    wPatch  = np.pad(wPatch, (((cnnx-nx)//2, (cnnx-nx)-(cnnx-nx)//2),(0,0),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            
                if ny>cnny:
                    rdfPatch = rdfPatch[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
                    maskPatch  = maskPatch[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
                    wPatch  = wPatch[:,(ny-cnny)//2:(ny-cnny)//2+cnny,:]
                elif ny<cnny:
                    rdfPatch = np.pad(rdfPatch, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                    maskPatch  = np.pad(maskPatch, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                    wPatch  = np.pad(wPatch, ((0,0),((cnny-ny)//2, (cnny-ny)-(cnny-ny)//2),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            
                if nz>cnnz:
                    rdfPatch = rdfPatch[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
                    maskPatch  = maskPatch[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
                    wPatch  = wPatch[:,:,(nz-cnnz)//2:(nz-cnnz)//2+cnnz]
                elif nz<cnnz:
                    rdfPatch  = np.pad(rdfPatch, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                    maskPatch  = np.pad(maskPatch, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))
                    wPatch  = np.pad(wPatch, ((0,0),(0,0),((cnnz-nz)//2, (cnnz-nz)-(cnnz-nz)//2)), 'constant', constant_values=((0,0),(0,0),(0,0)))

                rdfPatch = rdfPatch[np.newaxis, :]
                maskPatch = maskPatch[np.newaxis, :]
                wPatch = wPatch[np.newaxis, :] 
                
                # randomly load the perturbed field
                subdirs = glob(r"D:\Projects\QSM\datamix\fb_roi_96x96x96_qsm\*")
    
                rdir = subdirs[np.random.randint(len(subdirs), size=1)[0]]
                fmap_pertub = nib.load(os.path.join(rdir, 'fmap.nii.gz')).get_data()[np.newaxis,:]
                mask_pertub = nib.load(os.path.join(rdir, 'Mask.nii.gz')).get_data()[np.newaxis,:]
                suscp_pertub = nib.load(os.path.join(rdir, 'suscp.nii.gz')).get_data()[np.newaxis,:]               
    
                x1_list.append(100*rdfPatch*(maskPatch!=0))
                m1_list.append((maskPatch!=0))
                w1_list.append((wPatch)*(maskPatch!=0))
                k_list.append(kernel)
                
                x2_list.append(100*rdfPatch*(maskPatch!=0) + 100*fmap_pertub*(maskPatch!=0))
                y2_list.append(100/3.0*suscp_pertub*(maskPatch!=0))
                mroi_list.append(mask_pertub*(maskPatch!=0))
                mxroi_list.append(maskPatch*((maskPatch-mask_pertub)>0))
            
        return np.asarray(x1_list), np.asarray(m1_list), np.asarray(w1_list), np.asarray(k_list), np.asarray(x2_list), np.asarray(y2_list), np.asarray(mroi_list), np.asarray(mxroi_list)
